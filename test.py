import streamlit as st
import smtplib

# Define the names and credentials of the users
names = ["John", "Jane", "Jim"]
usernames = ["john123", "jane456", "jim789"]
passwords = ["pwJohn", "pwJane", "pwJim"]

# Prompt the user to select their name
selected_name = st.selectbox("Select your name:", names)

# Find the index of the selected name in the names list
index = names.index(selected_name)

# Retrieve the corresponding username and password
username = usernames[index]
password = passwords[index]

# Set up the email server and send the email to the user
if st.button("Send email"):
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login("aptest2177@gmail.com", "Routine77!")

        subject = "Username and Password"
        body = "Username: " + username + "\nPassword: " + password

        msg = f"Subject: {subject}\n\n{body}"
        server.sendmail("your_email@gmail.com", selected_name + "@gmail.com", msg)
        st.success("Email sent successfully.")
    except:
        st.error("Error: Email not sent.")
