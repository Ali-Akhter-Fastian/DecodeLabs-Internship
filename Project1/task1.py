
RESPONSES = {
    'hello':       'Hi there! How can I help you?',
    'hi':          'Hello! What can I do for you?',
    'how are you': 'I am just a bot, but I am running perfectly!',
    'what is ai':  'AI stands for Artificial Intelligence — machines that simulate human thinking.',
    'help':        'I can answer basic questions. Try: hello, how are you, what is ai, joke.',
    'joke':        'Why do programmers prefer dark mode? Because light attracts bugs!',
    'name':        'I am RuleBot, your rule-based AI assistant.',
}

def get_response(user_input):
    # Phase 1: Sanitization
    clean_input = user_input.lower().strip()

    # Phase 2: Exit strategy
    if clean_input in ('exit', 'quit', 'bye'):
        print("Bot: Goodbye! Have a great day.")
        return False

    # Phase 3 & 4: Dictionary lookup with fallback
    reply = RESPONSES.get(clean_input, "I do not understand that yet. Type 'help' to see what I know.")
    print(f"Bot: {reply}")
    return True

def main():
    print("Bot: Hello! I am RuleBot. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if not get_response(user_input):
            break

if __name__ == "__main__":
    main()
    