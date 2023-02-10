import streamlit as st
def main():
    st.title("User Management Page")

    username = st.text_input("Username")
    password = st.text_input("Password", type='password')

    if st.button("Submit"):
        if username == "admin" and password == "password":
            st.success("Login successful")
        else:
            st.error("Incorrect login or password")
    
    if st.checkbox("Reset username"):
        new_username = st.text_input("Enter new username")
        if st.button("Update username"):
            username = new_username
            st.success("Username updated successfully")
    
    if st.checkbox("Reset password"):
        new_password = st.text_input("Enter new password", type='password')
        confirm_password = st.text_input("Confirm new password", type='password')
        if st.button("Update password"):
            if new_password == confirm_password:
                password = new_password
                st.success("Password updated successfully")
            else:
                st.error("Password and confirmation do not match")

if __name__ == '__main__':
    main()
