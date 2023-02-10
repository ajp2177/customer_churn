import streamlit as st

def main():
    st.title("Login Page")

    username = st.text_input("Username")
    password = st.text_input("Password", type='password')

    if st.button("Submit"):
        if username == "admin" and password == "password":
            st.success("Login successful")
        else:
            st.error("Incorrect login or password")

if __name__ == '__main__':
    main()
