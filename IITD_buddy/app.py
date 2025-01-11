import streamlit as st
import openai  # Replace with your model's API library if different

# Set up OpenAI API key (replace with your API key or setup logic)
openai.api_key = "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"

# Streamlit app
st.title("IITD_buddy")

st.sidebar.header("Instructions")
st.sidebar.write("""
1. Enter your question in the text box below.
2. Click 'Generate Response' to see the answer from the AI model.
3. Adjust settings as needed in the sidebar.
""")

# User input
question = st.text_input("Ask your question:", placeholder="Type your question here...")
temperature = st.sidebar.slider("Response Creativity (Temperature)", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 300, 150)

# Generate response
if st.button("Generate Response"):
    if question.strip():
        try:
            # Call OpenAI API (adjust for your model)
            response = openai.Completion.create(
                engine="text-davinci-003",  # Replace with your preferred model
                prompt=question,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            answer = response.choices[0].text.strip()
            st.success("AI Response:")
            st.write(answer)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")

# Footer
st.sidebar.write("Built with Streamlit and OpenAI")