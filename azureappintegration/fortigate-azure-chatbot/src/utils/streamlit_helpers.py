def display_header(title):
    st.title(title)

def display_instructions(instructions):
    st.subheader("Instructions")
    st.write(instructions)

def get_user_input(prompt):
    return st.text_input(prompt)

def display_response(response):
    st.subheader("Response")
    st.write(response)

def display_error(message):
    st.error(message)