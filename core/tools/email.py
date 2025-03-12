import dspy
from typing import Any
from core.dspy_classes.email_parser import EmailParser
import smtplib
from email.message import EmailMessage
import getpass
from typing import Optional


class EmailTools(dspy.Module):
    """Enhanced email tool with multi-provider support and safety checks."""

    def __init__(self):
        super().__init__()
        self.email_parser = EmailParser()
        self.history = ''

    def _validate_fields(self, **fields) -> Optional[str]:
        """Validate required email fields."""
        missing = [k for k, v in fields.items() if not v]
        if missing:
            return f"Missing fields: {', '.join(missing)}. Please ask the user to provide them."
        return None

    def _get_smtp_config(self, email: str) -> tuple:
        """Get SMTP server based on email domain."""
        domain = email.split("@")[-1].lower()
        configs = {
            "qq.com": ("smtp.qq.com", 587),
            "gmail.com": ("smtp.gmail.com", 465),
            "outlook.com": ("smtp.office365.com", 587)
        }
        return configs.get(domain, (None, None))

    def modify_email_content(self, suggestion: str, parsed: Any) -> None:
        """Modify email content using DSPy based on user suggestion."""

        class EmailModifier(dspy.Signature):
            """Modify email content while preserving key information."""
            original_subject = dspy.InputField(desc="Current email subject line")
            original_body = dspy.InputField(desc="Current email body content")
            user_suggestion = dspy.InputField(desc="User's modification request")
            new_subject = dspy.OutputField(desc="Improved subject line")
            new_body = dspy.OutputField(desc="Revised email body content")

        predictor = dspy.ChainOfThought(EmailModifier)
        result = predictor(
            original_subject=parsed.subject or "",
            original_body=parsed.body or "",
            user_suggestion=suggestion
        )


        if result.new_subject:
            parsed.subject = result.new_subject
        if result.new_body:
            parsed.body = result.new_body



    def forward(self, query: str, internal_memory: dict) -> dspy.Prediction:
        user_input = input("Please enter the email information you want to send "
                                   "(sender's email address, password and name, receiver's email address and name, message subject and body), "
                                   "or omit it if it was included in the previous conversation:")
        self.history += f"\nUser query: {query + user_input}"

        parsed = self.email_parser(self.history).output

        while True:
            # Email Content Preview
            print("\n=== Current Email Draft ===")
            print(f"To: {parsed.receiver_name or 'Recipient'} <{parsed.receiver_email or 'unknown'}>")
            print(f"Subject: {parsed.subject or '[No Subject]'}")
            print("\nEmail Body Preview:")
            print(parsed.body or "[Empty content]")
            print("=" * 30 + "\n")


            modify_prompt = f"""Would you like to modify the email content? (yes/no) """
            modify = input(modify_prompt).lower()

            if modify == 'yes':
                suggestion = input("Please enter your modification suggestions (e.g., 'make the subject more professional', "
                                  "'shorten the body to 3 sentences'): ")
                self.modify_email_content(suggestion, parsed)
            else:
                break

        required_fields = {
            'sender_email': parsed.sender_email or getpass.getpass("Sender email (hidden): "),
            'sender_password': parsed.sender_password or getpass.getpass("Password (hidden): "),
            'receiver_email': parsed.receiver_email,
            'receiver_name': parsed.receiver_name,
            'body': parsed.body
        }

        if error_msg := self._validate_fields(**required_fields):
            return dspy.Prediction(result=error_msg, internal_result={})

        msg = EmailMessage()
        msg["Subject"] = parsed.subject or ""
        msg["From"] = f"{parsed.sender_name or 'Anonymous'} <{required_fields['sender_email']}>"
        msg["To"] = f"{parsed.receiver_name or 'Recipient'} <{required_fields['receiver_email']}>"
        msg.set_content(parsed.body)

        smtp_server, port = self._get_smtp_config(required_fields['sender_email'])
        if not smtp_server:
            return dspy.Prediction(
                result=f"Unsupported email provider: {required_fields['sender_email']}",
                internal_result={}
            )

        try:
            context = smtplib.SMTP_SSL if port == 465 else smtplib.SMTP
            with context(smtp_server, port) as server:
                if port == 587:  # TLS
                    server.starttls()
                server.login(required_fields['sender_email'], required_fields['sender_password'])
                server.send_message(msg)
                return dspy.Prediction(
                    result="Email sent successfully!",
                    internal_result={"email_sender_details": msg}
                )
        except Exception as e:
            return dspy.Prediction(
                result=f"Failed to send email: {str(e)}",
                internal_result={"email_sender_error": str(e)}
            )

