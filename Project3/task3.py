
# Pipeline:
#   1. Ingestion  - capture the user's skills (min. 3 required)
#   2. Scoring    - vectorize everything (TF-IDF) and compute cosine similarity
#   3. Sorting    - rank job roles by similarity score, descending
#   4. Filtering  - return the Top-N (3) most relevant roles

import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MIN_SKILLS_REQUIRED = 3
TOP_N = 3


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, "raw_skills.csv")


# SYNONYM / ALIAS MAP 

SKILL_ALIASES = {
    "ml": "machine_learning",
    "ai": "artificial_intelligence",  
    "dl": "deep_learning",
    "k8s": "kubernetes",
    "js": "javascript",
    "reactjs": "react",
    "node": "node_js",
    "nodejs": "node_js",
    "devops": "automation",
    "gcp": "cloud_computing",
    "cicd": "ci_cd",
}



# HELPER: normalize a skill so multi-part tags stay as ONE token
# (e.g. "Cloud Computing" -> "cloud_computing", "CI/CD" -> "ci_cd",
# "Node.js" -> "node_js"), then resolve any known alias to its
# canonical form.


def normalize_skill(skill: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", skill.strip().lower()).strip("_")
    return SKILL_ALIASES.get(normalized, normalized)


def build_skill_document(skills) -> str:
    return " ".join(normalize_skill(s) for s in skills)


# STEP 1: INGESTION - load the job-role dataset (the "items")

def load_job_roles(path=DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Skills"] = df["Skills"].apply(lambda s: [x.strip() for x in s.split(",")])
    df["SkillDoc"] = df["Skills"].apply(build_skill_document)
    return df



# STEP 1: INGESTION - capture the user's skills (min 3, per spec)

def get_user_skills() -> list:
    print("Enter at least 3 skills or interests, separated by commas.")
    print("Example: Python, Cloud Computing, Automation\n")

    while True:
        try:
            raw = input("Your skills: ")
        except EOFError:
            # no interactive input available -> fall back to the
            # example from the training deck so the script still runs
            print("(No input detected - using example: Python, Cloud Computing, Automation)")
            raw = "Python, Cloud Computing, Automation"

        skills = [s.strip() for s in raw.split(",") if s.strip()]
        if len(skills) >= MIN_SKILLS_REQUIRED:
            return skills
        print(f"Please enter at least {MIN_SKILLS_REQUIRED} skills "
              f"(you entered {len(skills)}). Try again.\n")



# STEP 2 & 3 & 4: SCORING, SORTING, FILTERING

def recommend_careers(user_skills, job_df, top_n=TOP_N):
    # Build the shared vocabulary from the item corpus only, then map
    # the user profile into that SAME vector space (this is what makes
    # the comparison mathematically valid).
    vectorizer = TfidfVectorizer()
    item_vectors = vectorizer.fit_transform(job_df["SkillDoc"])

    user_doc = build_skill_document(user_skills)
    user_vector = vectorizer.transform([user_doc])

    scores = cosine_similarity(user_vector, item_vectors).flatten()

    results = job_df.copy()
    results["MatchScore"] = scores
    results = results.sort_values("MatchScore", ascending=False)

    # Cold-start guard: if every score is 0, the user's skills share no
    # vocabulary with any job role -> fall back to popular/general roles
    # instead of returning a meaningless "Top 3 of nothing".
    if results["MatchScore"].max() == 0:
        print("\nNo direct skill overlap found with any job role.")
        print("Showing general/popular roles as a starting point instead:\n")
        return results.head(top_n), False

    return results.head(top_n), True


def display_recommendations(user_skills, recommendations, matched):
    user_set = {normalize_skill(s) for s in user_skills}
    print("\n" + "=" * 55)
    print("TOP CAREER RECOMMENDATIONS")
    print("=" * 55)

    for rank, (_, row) in enumerate(recommendations.iterrows(), start=1):
        role_skills = {normalize_skill(s) for s in row["Skills"]}
        overlap = sorted(user_set & role_skills)
        overlap_display = ", ".join(overlap) if overlap else "-"
        match_pct = row["MatchScore"] * 100

        print(f"\n#{rank}  {row['JobRole']}")
        print(f"    Match Score : {match_pct:.1f}%")
        print(f"    Overlapping Skills : {overlap_display}")
        print(f"    Full Skill Set     : {', '.join(row['Skills'])}")


def main():
    job_df = load_job_roles()
    user_skills = get_user_skills()
    print(f"\nSkills received: {', '.join(user_skills)}")

    recommendations, matched = recommend_careers(user_skills, job_df, TOP_N)
    display_recommendations(user_skills, recommendations, matched)


if __name__ == "__main__":
    main()
