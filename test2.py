from app import get_tax_assistant
import sys
import time

def main():
    assistant = get_tax_assistant()

    question = "Explain VAT in Nigeria in simple terms"

    print("\nAssistant:", end=" ", flush=True)

    try:
        for chunk in assistant.ask_question(question, user_id="terminal-test"):
            # Some chunks may repeat full content depending on model,
            # so we print only the new text
            sys.stdout.write(chunk)
            sys.stdout.flush()
            time.sleep(0.01)  # smooth typing effect (optional)

        print("\n\n[Stream finished]")

    except Exception as e:
        print("\nError:", str(e))


if __name__ == "__main__":
    main()
