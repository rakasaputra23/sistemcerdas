import pickle
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Memuat model yang sudah disimpan
model = pickle.load(open('model_prediksi_harga_mobil.sav', 'rb'))

# Judul aplikasi
st.title('Prediksi Harga Mobil')

# Menampilkan header untuk dataset
st.header("Dataset Mobil")

# Membaca dataset dan menampilkannya
df1 = pd.read_csv('CarPrice.csv')
st.dataframe(df1)

# Menampilkan beberapa grafik untuk fitur yang relevan
st.write("Grafik Highway-mpg")
chart_highwaympg = pd.DataFrame(df1, columns=["highwaympg"])
st.line_chart(chart_highwaympg)

st.write("Grafik Curbweight")
chart_curbweight = pd.DataFrame(df1, columns=["curbweight"])
st.line_chart(chart_curbweight)

st.write("Grafik Horsepower")
chart_horsepower = pd.DataFrame(df1, columns=["horsepower"])
st.line_chart(chart_horsepower)

# Input nilai untuk prediksi
st.header("Masukkan Data Mobil")

# Input nilai untuk fitur-fitur mobil
highwaympg = st.number_input('Highway MPG', min_value=0, max_value=100, value=30)
curbweight = st.number_input('Curbweight (kg)', min_value=0, max_value=5000, value=2000)
horsepower = st.number_input('Horsepower', min_value=0, max_value=500, value=100)

# Menampilkan tombol untuk melakukan prediksi
if st.button('Prediksi'):
    # Membuat DataFrame dari input untuk prediksi
    input_data = pd.DataFrame([[highwaympg, curbweight, horsepower]], columns=['highwaympg', 'curbweight', 'horsepower'])
    
    # Melakukan prediksi menggunakan model yang dimuat
    car_prediction = model.predict(input_data)
    
    # Mengubah hasil prediksi menjadi angka desimal
    harga_mobil_str = np.array(car_prediction)
    harga_mobil_float = float(harga_mobil_str[0])

    # Menampilkan hasil prediksi
    harga_mobil_formatted = "${:,.2f}".format(harga_mobil_float)
    st.write(f"Harga mobil yang diprediksi adalah: {harga_mobil_formatted}")

# Menambahkan grafik distribusi harga mobil
st.write("Distribusi Harga Mobil")
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(df1['price'], kde=True, color='blue')
plt.title("Distribusi Harga Mobil")
plt.xlabel("Price")
plt.ylabel("Frequency")
st.pyplot()

# Menambahkan informasi tentang aplikasi
st.sidebar.header("Tentang Aplikasi")
st.sidebar.write("""
Aplikasi ini digunakan untuk memprediksi harga mobil berdasarkan fitur-fitur tertentu seperti highway mpg, curbweight, dan horsepower.
Model ini telah dilatih menggunakan dataset mobil dan kemudian disimpan untuk digunakan dalam aplikasi web ini.
""")