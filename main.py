from datetime import datetime
import streamlit as st
import imaplib
import os
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import tiktoken
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
import logging

load_dotenv()

# Set up logging
logging.basicConfig(
    filename="email_query_assistant.log",  # Log file name
    level=logging.INFO,  # Set to DEBUG for detailed logs, change to INFO or WARNING for less verbose logging
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("Application started.")
# Initialize OpenAI client
client = OpenAI()

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(),
)
collection = chroma_client.get_or_create_collection("emails")
logging.info("ChromaDB client initialized and collection created.")


def fetch_emails(imap_host, max_emails=None):
    """
    Fetch the most recent `max_emails` from the inbox based on the received date.
    """
    logging.info("Connecting to email server.")
    try:
        mail = imaplib.IMAP4_SSL(imap_host)
        mail.login(
            os.getenv("EMAIL"),
            os.getenv("PASSWORD"),
        )
        mail.select("inbox")

        # Search for all email IDs
        logging.info("Fetching email IDs from inbox.")
        status, messages = mail.search(None, "ALL")
        email_ids = messages[0].split()

        emails = []
        for e_id in email_ids:
            # Fetch the raw email data
            status, data = mail.fetch(e_id, "(RFC822)")
            raw_email = data[0][1]

            # Parse the email to extract metadata
            email_message = BytesParser().parsebytes(raw_email)
            received_date = parsedate_to_datetime(email_message["Date"])
            subject = email_message["Subject"] or "No Subject"
            sender = email_message["From"] or "Unknown"

            # Extract the email body
            body = None
            if email_message.is_multipart():
                for part in email_message.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition", ""))
                    if (
                        content_type == "text/plain"
                        and "attachment" not in content_disposition
                    ):
                        body = part.get_payload(decode=True).decode(
                            "utf-8", errors="ignore"
                        )
                        break  # Prefer plain text over other formats
                    elif content_type == "text/html" and not body:  # Fallback to HTML
                        body = BeautifulSoup(
                            part.get_payload(decode=True), "html.parser"
                        ).get_text(strip=True)
            else:
                content_type = email_message.get_content_type()
                if content_type == "text/plain":
                    body = email_message.get_payload(decode=True).decode(
                        "utf-8", errors="ignore"
                    )
                elif content_type == "text/html":
                    body = BeautifulSoup(
                        email_message.get_payload(decode=True), "html.parser"
                    ).get_text(strip=True)

            body = body or "No Body"  # Default if no body is found

            emails.append(
                {
                    "subject": subject,
                    "body": body,
                    "sender": sender,
                    "date": received_date,
                }
            )

        # Sort emails by received date in descending order
        emails.sort(key=lambda x: x["date"], reverse=True)

        # Log all fetched emails
        logging.info(f"Fetched {len(emails)} emails.")
        for idx, email in enumerate(emails):
            logging.info(
                f"Email {idx + 1}: Subject: {email['subject']}, "
                f"Sender: {email['sender']}, Date: {email['date']}, "
                f"Body: {email['body'][:100]}..."  # Log only the first 100 characters of the body for brevity
            )

        logging.info(f"Returning the most recent {max_emails} emails.")
        return emails[:max_emails]

    except Exception as e:
        logging.error(f"Error fetching emails: {str(e)}", exc_info=True)
        raise


# Tokenizer for the model
encoding = tiktoken.encoding_for_model("text-embedding-3-small")


def truncate_text(text, max_tokens=8192):
    try:
        tokens = encoding.encode(text)
        if len(tokens) > max_tokens:
            logging.debug(
                f"Truncating text: {len(tokens)} tokens exceed the limit of {max_tokens}."
            )
            tokens = tokens[:max_tokens]
        return encoding.decode(tokens)
    except Exception as e:
        logging.error(f"Error truncating text..{str(e)}", exc_info=True)
        raise


def get_embeddings(text):
    try:
        # Truncate text to fit the token limit
        truncated_text = truncate_text(text, max_tokens=8192)
        logging.debug("Generating embeddings for text.")
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=truncated_text,
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error generating embeddings.{str(e)}", exc_info=True)
        raise


def store_emails(emails):
    try:
        logging.info(f"Storing {len(emails)} emails in ChromaDB.")
        for idx, email in enumerate(emails):
            text = (
                f"{email['subject']} {email['body']} {email['date']} {email['sender']}"
            )
            text = truncate_text(text, max_tokens=8192)
            embedding = get_embeddings(text)
            metadata = {
                "subject": email["subject"],
                "body": email["body"],  # Include body in metadata
                "sender": email["sender"],
                "date": (
                    email["date"].isoformat()
                    if isinstance(email["date"], datetime)
                    else email["date"]
                ),
            }
            collection.add(
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[str(idx)],
            )
        logging.info("Emails stored successfully.")
    except Exception as e:
        logging.error(f"Error storing emails in ChromaDB.{str(e)}", exc_info=True)
        raise


# Search emails
def search_emails(query, top_k=10):
    try:
        logging.info(f"Searching for emails with query: {query}.")
        query_embedding = get_embeddings(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        logging.info(f"Result {results} results.")
        # Log all the metadata for the searched emails
        if results["metadatas"]:
            logging.info(
                f"Search completed. Found {len(results['metadatas'][0])} results."
            )
            for idx, metadata in enumerate(results["metadatas"][0]):
                logging.info(f"Email {idx + 1}: {metadata}")
        else:
            logging.info("Search completed. No results found.")

        return results["metadatas"][0] if results["metadatas"] else []
    except Exception as e:
        logging.error(f"Error during email search: {str(e)}", exc_info=True)
        raise


def generate_response(query, emails):
    try:
        logging.info(
            f"Generating response for query: {query} with {len(emails)} emails."
        )
        # Ensure emails is a list of dictionaries
        context = "\n\n".join(
            [
                f"Email {i+1}: Subject: '{email.get('subject', 'No Subject')}', Body: '{email.get('body', 'No Body')}', Date: {email.get('date', 'No Date')}, Sender: {email.get('sender', 'No Sender')}"
                for i, email in enumerate(emails)
            ]
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an advanced email assistant with the ability to analyze multiple email threads. "
                        "Your responses should be concise, informative, and structured. When answering, "
                        "provide insights, identify key themes, and highlight any specific details that address the user's query."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Based on the following emails, respond to the query in detail and with clarity:\n\n{context}\n\n"
                        f"Query: {query}\n\n"
                        "Please ensure your response is comprehensive, includes relevant details, and offers actionable insights."
                    ),
                },
            ],
        )
        logging.info("Response generated successfully.")
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating response.{str(e)}", exc_info=True)
        raise


# Streamlit UI
st.title("Email Query Assistant")
logging.info("Streamlit UI initialized.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []  # Store chat messages

with st.sidebar:
    st.subheader("Settings")
    st.write("Ensure your emails are stored in ChromaDB before using this app.")
    if st.button("Fetch & Store Emails"):
        logging.info("Fetch & Store Emails button clicked.")
        emails = fetch_emails(
            "imap.gmail.com",
            # os.getenv("EMAIL"),
            # os.getenv("PASSWORD"),
            # max_emails=10,
        )
        store_emails(emails)
        st.success("Emails fetched and stored successfully!")
        logging.info("Emails fetched and stored successfully.")

# Display chat history
for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.chat_message("user").markdown(message["content"])
    else:
        st.chat_message("assistant").markdown(message["content"])

# User input
if user_input := st.chat_input("Ask your query about emails"):
    logging.info(f"User query received: {user_input}")
    # Add user message to session state
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    # Query emails and generate response
    email_results = search_emails(user_input)
    logging.info(f"Email search results: {email_results}")
    response = generate_response(user_input, email_results)

    # Add assistant response to session state
    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.chat_message("assistant").markdown(response)
    logging.info("Response sent to user.")


# # Fetch emails
# def fetch_emails(imap_host, email, password):
#     mail = imaplib.IMAP4_SSL(imap_host)
#     mail.login(email, password)
#     mail.select("inbox")
#     status, messages = mail.search(None, "ALL")
#     email_ids = messages[0].split()
#     emails = []
#     for e_id in email_ids:
#         status, data = mail.fetch(e_id, "(RFC822)")
#         raw_email = data[0][1].decode("utf-8")
#         soup = BeautifulSoup(raw_email, "html.parser")
#         emails.append(
#             {
#                 "subject": (
#                     soup.find("subject").get_text(strip=True)
#                     if soup.find("subject")
#                     else "No Subject"
#                 ),
#                 "body": soup.get_text(strip=True),
#                 "sender": (
#                     soup.find("from").get_text(strip=True)
#                     if soup.find("from")
#                     else "Unknown"
#                 ),
#                 "date": (
#                     soup.find("date").get_text(strip=True)
#                     if soup.find("date")
#                     else "Unknown"
#                 ),
#             }
#         )
#     return emails
