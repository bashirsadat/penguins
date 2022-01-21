import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
import altair as alt
from sklearn.ensemble import RandomForestClassifier
image = Image.open('logo.png')
cola, colb, colc = st.columns([3,6,1])
with cola:
    st.write("")

with colb:
    st.image(image, width = 300)

with colc:
    st.write("")
st.write("""
# Penguin Prediction App

This app predicts the **Palmer Penguin** species!

Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")
cola, colb, colc = st.columns([1,6,1])
irisi = Image.open('all.png')
with cola:
    st.write("")
with colb:
    st.image(irisi, width = 600)
with colc:
    st.write("")

menu = ["Home","About"]
choice = st.sidebar.selectbox("Menu",menu)
if choice == "Home":
    st.sidebar.header('User Input Parameters')


    st.sidebar.header('User Input Features')

    st.sidebar.markdown("""
    [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
    """)

    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
            sex = st.sidebar.selectbox('Sex',('male','female'))
            bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
            bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
            flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
            body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
            data = {'island': island,
                    'bill_length_mm': bill_length_mm,
                    'bill_depth_mm': bill_depth_mm,
                    'flipper_length_mm': flipper_length_mm,
                    'body_mass_g': body_mass_g,
                    'sex': sex}
            features = pd.DataFrame(data, index=[0])
            return features
        input_df = user_input_features()

    # Combines user input features with entire penguins dataset
    # This will be useful for the encoding phase
    penguins_raw = pd.read_csv('penguins_cleaned.csv')
    penguins = penguins_raw.drop(columns=['species'])
    df = pd.concat([input_df,penguins],axis=0)

    # Encoding of ordinal features
    # https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
    encode = ['sex','island']
    for col in encode:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df,dummy], axis=1)
        del df[col]
    df = df[:1] # Selects only the first row (the user input data)

    # Displays the user input features
    st.subheader('User Input features')

    if uploaded_file is not None:
        st.write(df)
    else:
        st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
        st.write(df)

    # Reads in saved classification model
    load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

    # Apply model to make predictions
    prediction = load_clf.predict(df)
    prediction_proba = load_clf.predict_proba(df)


    st.subheader('Prediction')
    penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
    # st.write(penguins_species[prediction])
    cola, colb, colc = st.columns([4,6,1])
    Gentoo = Image.open('gentoo.jpg')
    Chinstrap = Image.open('chinstrap.jpg')
    Adelie = Image.open('adelie.jpg')
    with cola:
        st.write("")
    with colb:
        if (penguins_species[prediction]=="Adelie"):
            st.image(Adelie, width = 200)
            st.write("Adelie")
        elif (penguins_species[prediction]=="Chinstrap"):
            st.image(Chinstrap, width = 200)
            st.write("Chinstrap")
        elif (penguins_species[prediction]=="Gentoo"):
            st.image(Gentoo, width = 200)
            st.write("Gentoo")
    with colc:
        st.write("")


    st.subheader('Prediction Probability')
    # st.write(prediction_proba)

    proba_df_clean = prediction_proba.T
    proba_df= pd.DataFrame(proba_df_clean, columns=["Probabilities"])
    penguins_n= ['Adelie','Chinstrap','Gentoo']
    proba_df["Penguins"]= penguins_n
    # st.write(type(proba_df))
    column_names = ["Penguins", "Probabilities"]
    proba_df = proba_df.reindex(columns=column_names)
    st.write(proba_df)
    fig = alt.Chart(proba_df).mark_bar().encode(x='Penguins',y='Probabilities',color='Penguins')
    st.altair_chart(fig,use_container_width=True)
else:
    st.subheader("About")
    st.write("With a hybrid profile of data science and computer science, I’m pursuing a career in AI-driven firms. I believe in dedication, discipline, and creativity towards my job, which will be helpful in meeting your firm's requirements as well as my personal development.")
    st.write("Check out this project's [Github](https://github.com/bashirsadat/penguins)")
    st.write(" My [Linkedin](https://www.linkedin.com/in/saadaat/)")
    st.write("See my other projects [LinkTree](https://linktr.ee/saadaat)")
