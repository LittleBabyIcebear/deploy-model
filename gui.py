# app_streamlit.py
import streamlit as st
import requests
from PIL import Image
import io

def predict_image(image_file):
    # API endpoint
    url = "http://10.3.148.65:5000/predict"
    
    files = {'file': image_file}
    
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()  
        
        results = response.json()
        return results
    except requests.exceptions.RequestException as e:
        st.error(f"Koneksi API Eror: {str(e)}")
        return None

def main():
    st.title("COVID-19 X-Ray Classification")
    st.write("Upload X-ray image")
    
    # File uploader
    uploaded_file = st.file_uploader("Pilih Image dari device...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        if st.button("Predict"):
            with st.spinner("Make prediction..."):

                uploaded_file.seek(0)
                results = predict_image(uploaded_file)
                
                if results:
                    with col2:
                        st.subheader("Prediction Results")
                        
                        # Display prediction 
                        st.markdown(f"<h3 style='color: #1f77b4;'>{results['prediction']}</h3>", 
                                  unsafe_allow_html=True)
                        
                        st.write("Confidence Score:")
                        st.progress(results['confidence'])
                        st.write(f"{results['confidence']*100:.2f}%")
                        
                        # Display probabilities
                        st.write("\nClass Probabilities:")
                        for class_name, prob in results['probabilities'].items():
                            st.write(f"{class_name}: {prob*100:.2f}%")

if __name__ == "__main__":
    main()