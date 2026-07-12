import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

# 1. LOAD AND UNDERSTAND THE DATASET
df = pd.read_excel("Dataset for Data Analytics.xlsx")

print("Dataset shape:", df.shape)
print("\nColumn info:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget (OrderStatus) distribution:\n", df["OrderStatus"].value_counts())

# CouponCode has some missing values for orders where no coupon was used, so it's a real
# category, not missing data -> fill it instead of dropping rows (I fill it with "NoCoupon" to indicate no coupon was used)
df["CouponCode"] = df["CouponCode"].fillna("NoCoupon")

# 2. SELECT FEATURES (x) AND TARGET (y)

# OrderID, CustomerID, TrackingNumber, ShippingAddress and Date are just
# identifiers / free text with no real predictive pattern, and TotalPrice
# is simply Quantity * UnitPrice, so it doesn't add new information.
# We drop all of these to keep only genuinely useful features.

x = df.drop(columns=["OrderID", "CustomerID", "TrackingNumber",
                      "ShippingAddress", "Date", "TotalPrice", "OrderStatus"])
y = df["OrderStatus"]

num_cols = ["Quantity", "UnitPrice", "ItemsInCart"]
cat_cols = ["Product", "PaymentMethod", "CouponCode", "ReferralSource"]

# 3. TRAIN-TEST SPLIT 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# 4. PREPROCESSING: SCALE NUMERIC FEATURES, ENCODE CATEGORICAL FEATURES

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# fit ONLY on training data, then just transform test data -> no data leakage

x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# 5. CHOOSE A GOOD VALUE OF K (elbow method)

print("\nChecking accuracy for different values of K:")
k_values = [1, 3, 5, 7, 9, 11, 15, 20]
best_k, best_acc = None, 0

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    acc = accuracy_score(y_test, model.predict(x_test))
    print(f"  k={k:<3} -> accuracy = {acc:.4f}")
    if acc > best_acc:
        best_k, best_acc = k, acc

print(f"\nBest K = {best_k} (accuracy = {best_acc:.4f})")


# 6. TRAIN FINAL MODEL

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)


# 7. EVALUATE THE MODEL (accuracy alone isn't enough - check F1 & confusion matrix too)

print("\nAccuracy:", accuracy_score(y_test, y_pred) * 100, "%")
print("F1 Score (weighted):", f1_score(y_test, y_pred, average="weighted"))

print("\nConfusion Matrix:")
print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                    index=le.classes_, columns=le.classes_))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
