# import necessary libraries
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import cloudpickle
import shap

# load model
with open("model.pkl", "rb") as f:
    pipeline = cloudpickle.load(f)
model = pipeline.named_steps['model']
preprocessor = pipeline.named_steps['preprocessor']

# mapping
product_category_mapping = {
    'bed_bath_table': 0,
    'computers_accessories': 1,
    'consoles_games': 2,
    'cool_stuff': 3,
    'furniture_decor': 4,
    'garden_tools': 5,
    'health_beauty': 6,
    'perfumery': 7,
    'watches_gifts': 8
}
month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

# Streamlit
st.title("Retail Price predictor")
if "page" not in st.session_state:
    st.session_state.page = "predict"

# user input
st.subheader("Product details")
product_category_name = st.selectbox("category", ['bed_bath_table', 'computers_accessories', 'consoles_games', 'cool_stuff', 'furniture_decor', 'garden_tools', 'health_beauty','perfumery', 'watches_gifts'])
product_weight_g = st.number_input("weight", min_value= 0.0, format= '%.2f')
volume = st.number_input("volume", min_value= 0)
st.markdown("---")

st.subheader("Date")
month = st.selectbox("month", ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
weekday = st.selectbox("no of weekdays", [20, 21, 22, 23])
weekend = st.selectbox("no of weekends", [8, 9, 10])
holiday = st.number_input("no of holidays", min_value= 0, max_value= 31)
st.markdown("---")

st.subheader("Your store data")
product_score = st.number_input("customer satisfaction", min_value= 0.0, max_value= 5.0, format= '%.1f', step= 0.1)
customers = st.number_input("no of customer", min_value= 0)
unit_price = st.number_input("price", min_value= 0.0, format= '%.2f')
qty = st.number_input("quantity", min_value= 1)
freight_price = st.number_input("shipping cost", min_value= 0.0, format= '%.2f')
sales_rate = st.number_input("sales rate", min_value= 0.0, max_value= 100.0, format= '%.2f')
st.markdown("---")

st.subheader("competitor store 1 data")
ps1 = st.number_input("customer satisfaction", min_value= 0.0, max_value= 5.0, format= '%.1f', step= 0.1, key= 'ps1')
comp_1 = st.number_input("price", min_value= 0.0, format= '%.2f', key= 'comp_1')
fp1 = st.number_input("shipping cost", min_value= 0.0, format= '%.2f', key= 'fp1')
st.markdown("---")

st.subheader("competitor store 2 data")
ps2 = st.number_input("customer satisfaction", min_value= 0.0, max_value= 5.0, format= '%.1f', step= 0.1, key= 'ps2')
comp_2 = st.number_input("price", min_value= 0.0, format= '%.2f', key= 'comp_2')
fp2 = st.number_input("shipping cost", min_value= 0.0, format= '%.2f', key= 'fp2')
st.markdown("---")

st.subheader("competitor store 3 data")
ps3 = st.number_input("customer satisfaction", min_value= 0.0, max_value= 5.0, format= '%.1f', step= 0.1, key= 'ps3')
comp_3 = st.number_input("price", min_value= 0.0, format= '%.2f', key= 'comp_3')
fp3 = st.number_input("shipping cost", min_value= 0.0, format= '%.2f', key= 'fp3')
st.markdown("---")

# prediction
if st.button("submit"):
    df = pd.DataFrame({
        "product_category_name": product_category_mapping[product_category_name],
        "product_weight_g": product_weight_g,
        "volume": volume,
        "month": month_mapping[month],
        "weekday": weekday,
        "weekend": weekend,
        "holiday": holiday,
        "product_score": product_score,
        "customers": customers,
        "unit_price": unit_price,
        "qty": qty,
        "freight_price": freight_price,
        "sales_rate": sales_rate,
        "ps1": ps1,
        "comp_1": comp_1,
        "fp1": fp1,
        "ps2": ps2,
        "comp_2": comp_2,
        "fp2": fp2,
        "ps3": ps3,
        "comp_3": comp_3,
        "fp3": fp3
    }, index=[0])
    prediction = pipeline.predict(df)[0]
    st.text(f"predicted price: {prediction}")
    st.session_state.page = "explain"
    st.session_state.df = df
    st.session_state.prediction = prediction
    st.session_state.columns_no = len(df.columns.tolist())

# shap explainer
if st.session_state.page == "explain":
    prediction = st.session_state.prediction
    st.text(f"predicted price: {prediction}")
    with st.form("Explainer form"):
        top_k = st.number_input("no of factors", min_value= 1, max_value= st.session_state.columns_no)
        if st.form_submit_button("explain"):
            explainer = shap.TreeExplainer(model)
            df = preprocessor.transform(st.session_state.df)
            shap_values = explainer.shap_values(df)

            # force plot
            st.subheader("SHAP Force Plot")
            force_plot = shap.force_plot(explainer.expected_value, shap_values, df, matplotlib=True)
            st.pyplot(plt.gcf())

            # dropout explainer
            def shap_explanation(top_k):
                """
                Returns a textual SHAP explanation for a single prediction.
                
                Parameters:
                - top_k : number of top contributing features to show
                
                Returns:
                - explanation : str
                """
                
                # Map features to SHAP values
                features = st.session_state.df.columns
                top_features = sorted(zip(features, shap_values[0]), key=lambda t: abs(t[1]), reverse=True)[:top_k]
                
                # Build explanation text
                explainer = f"Explaination:\n"
                for name, val in top_features:
                    direction = "increased" if val > 0 else "decreased"
                    explainer += f"- {name} {direction} the risk by {val:+.3f}\n"

                return explainer


            st.text(shap_explanation(top_k))

