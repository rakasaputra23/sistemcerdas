import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Set page configuration at the start
st.set_page_config(page_title="Prediksi Harga Rumah", layout="wide")

# Load the dataset
df = pd.read_excel('DATA RUMAH.xlsx')

# Rename columns for clarity
df = df.rename(columns={
    'NO': 'nomor',
    'NAMA RUMAH': 'nama_rumah',
    'HARGA': 'harga',
    'LB': 'lb',
    'LT': 'lt',
    'KT': 'kt',
    'KM': 'km',
    'GRS': 'grs'
})

# Convert price to millions for easier interpretation
df['harga'] = (df['harga']/1000000).astype(int)

# Drop 'nomor' column as it's not needed for prediction
df.drop(columns=['nomor'], inplace=True)

# Classification of price into categories for analysis
q1 = df['harga'].quantile(0.25)
median = df['harga'].median()
q3 = df['harga'].quantile(0.75)

def classification_harga(harga):
    if harga <= q1:
        return 'Murah'
    elif harga <= median:
        return 'Menengah'
    else:
        return 'Mahal'

# Add 'tingkat_harga' column for price classification
df['tingkat_harga'] = df['harga'].apply(classification_harga)

# Home Page with Deskripsi Aplikasi and Machine Learning Explanation
def show_home():
    st.title("Selamat Datang di Aplikasi Prediksi Harga Rumah")
    
    # Display an image on the home page
    st.image('rumah.jpg', use_container_width=True)

    st.markdown(""" 
    Aplikasi ini dibuat untuk membantu Anda memprediksi harga rumah berdasarkan beberapa fitur rumah. 
    Dengan menggunakan model **Machine Learning**, aplikasi ini memanfaatkan data historis harga rumah 
    dan fitur-fitur seperti luas bangunan, luas tanah, jumlah kamar tidur, kamar mandi, dan garasi untuk memprediksi harga rumah.

    ### Tentang Machine Learning
    Machine learning (ML) adalah cabang dari kecerdasan buatan (AI) yang memungkinkan komputer untuk belajar 
    dari data dan membuat keputusan tanpa perlu diprogram secara eksplisit. 
    Dalam konteks aplikasi ini, model **Linear Regression** digunakan untuk memprediksi harga rumah berdasarkan fitur-fitur yang ada.
    
    ### Tujuan Aplikasi
    Tujuan utama dari aplikasi ini adalah untuk memberikan perkiraan harga rumah berdasarkan fitur-fitur yang relevan, 
    seperti luas bangunan, luas tanah, jumlah kamar tidur, dan lainnya. Dengan aplikasi ini, pengguna dapat mengetahui harga 
    rumah yang sesuai dengan kriteria mereka, baik itu untuk membeli, menjual, atau investasi properti.
    
    ### Pembuat Aplikasi
    Aplikasi ini dibuat oleh KELOMPOK 4 dalam bidang data science dan machine learning.
    Pembuat aplikasi ini memiliki pengalaman dalam membangun model prediktif untuk membantu pengambilan keputusan berbasis data.
    
    ### 1. **Judul**:
    **Prediksi Harga Rumah Menggunakan Machine Learning** 

    ### 2. **Sumber Dataset + Alasan**:
    Dataset yang digunakan dalam aplikasi ini diperoleh dari dataset **DATA RUMAH.xlsx**. Dataset ini memuat informasi tentang harga rumah yang dipengaruhi oleh berbagai faktor seperti luas bangunan, luas tanah, jumlah kamar tidur, kamar mandi, dan garasi. Dataset ini relevan karena memberikan gambaran harga rumah yang dapat digunakan untuk membangun model prediktif harga rumah.

    ### 3. **Exploratory Data Analysis (EDA)**:
    - **Pre-processing Data**: Sebelum membangun model, data terlebih dahulu diproses untuk memastikan kebersihan dan keteraturan data. Kolom yang tidak relevan, seperti nomor rumah, dihapus. Kolom harga juga dikonversi ke satuan juta agar lebih mudah dipahami.
    - **Pengecekan Missing Values**: Memeriksa apakah ada nilai yang hilang di dataset dan menangani masalah tersebut.
    - **Klasifikasi Harga**: Menggunakan pembagian kuartil untuk mengklasifikasikan harga rumah menjadi kategori "Murah", "Menengah", dan "Mahal".

    ### 4. **Visualisasi Data Utama**:
    Data dianalisis lebih lanjut dengan visualisasi menggunakan histograms dan heatmap untuk menunjukkan distribusi harga rumah dan korelasi antara fitur-fitur rumah.

    ### 5. **Modeling ML dan Metode yang Digunakan**:
    - **Metode**: Dalam aplikasi ini, model **Linear Regression** digunakan untuk memprediksi harga rumah berdasarkan fitur seperti luas bangunan, luas tanah, jumlah kamar tidur, kamar mandi, dan garasi.
    - **Evaluasi Model**: Model dievaluasi menggunakan metrik MAE (Mean Absolute Error), MSE (Mean Squared Error), dan R² score untuk mengevaluasi kualitas prediksi.

    ### 6. **Terima kasih telah menggunakan aplikasi ini!**
    """)

# Function to show sidebar and navigation with modern dropdown
def show_sidebar():
    st.sidebar.title("Navigasi")
    options = st.sidebar.selectbox("Pilih Halaman", ['Home', 'Prediksi Harga Rumah', 'Analisis Data', 'Evaluasi Model', 'Insight Model'])
    return options

# Display data function
def show_data():
    st.header("Data Rumah")
    st.write("Menampilkan beberapa baris pertama dari dataset:") 
    st.write(df.head())
    st.write(""" 
    Dataset ini berisi informasi tentang harga rumah yang dapat digunakan untuk memprediksi harga berdasarkan fitur-fitur seperti luas bangunan, luas tanah, jumlah kamar tidur, kamar mandi, dan garasi.
    """)

# Show price distribution
def show_price_distribution():
    st.header("Distribusi Harga Rumah")
    plt.figure(figsize=(8, 6))
    sns.histplot(df['harga'], kde=True, color='blue')
    plt.title('Distribusi Harga Rumah')
    plt.xlabel('Harga Rumah')
    plt.ylabel('Frekuensi')
    st.pyplot(plt)
    plt.clf()
    st.write(""" 
    Distribusi harga rumah menunjukkan sebaran harga pada dataset ini. Dengan analisis distribusi, kita dapat mengetahui pola harga rumah, apakah terdapat konsentrasi harga tertentu atau harga yang lebih tersebar merata.
    """)

# Show correlation matrix
def show_correlation():
    st.header("Korelasi antara Fitur dan Harga Rumah")
    df_corr = df.drop(['tingkat_harga', 'nama_rumah'], axis=1)
    correlation_all = df_corr.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_all, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Korelasi Fitur dengan Harga Rumah')
    st.pyplot(plt)
    plt.clf()
    st.write(""" 
    Matriks korelasi menunjukkan hubungan antara fitur-fitur yang ada dengan harga rumah. Nilai korelasi yang mendekati 1 atau -1 menunjukkan hubungan yang kuat antara fitur tersebut dengan harga rumah. Korelasi positif berarti fitur tersebut meningkat seiring dengan harga, sedangkan korelasi negatif berarti fitur tersebut menurun seiring dengan harga.
    """)

# Train and evaluate model
def train_and_evaluate_model():
    # Data preprocessing
    X = df[['lb', 'lt', 'kt', 'km', 'grs']].values  # Features
    y = df['harga'].values  # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, 'model_prediksi_harga_rumah.pkl')

    # Evaluate model on training and testing data
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    mae_train = mean_absolute_error(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)

    mae_test = mean_absolute_error(y_test, y_pred)
    mse_test = mean_squared_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)

    st.subheader("Evaluasi Model")
    st.write("Evaluasi Model pada Data Pelatihan:")
    st.write(f'MAE (Mean Absolute Error): {mae_train}')
    st.write(f'MSE (Mean Squared Error): {mse_train}')
    st.write(f'R2 Score: {r2_train}')
    st.write("\nEvaluasi Model pada Data Pengujian:")
    st.write(f'MAE: {mae_test}')
    st.write(f'MSE: {mse_test}')
    st.write(f'R2 Score: {r2_test}')

# Function to predict house price
def show_predict():
    st.header('Masukkan Fitur Rumah')
    slider_lb = st.slider('Luas Bangunan (m²)', min_value=int(df['lb'].min()), max_value=int(df['lb'].max()), value=100, step=10)
    slider_lt = st.slider('Luas Tanah (m²)', min_value=int(df['lt'].min()), max_value=int(df['lt'].max()), value=300, step=10)
    slider_kt = st.slider('Jumlah Kamar Tidur', min_value=int(df['kt'].min()), max_value=int(df['kt'].max()), value=3, step=1)
    slider_km = st.slider('Jumlah Kamar Mandi', min_value=int(df['km'].min()), max_value=int(df['km'].max()), value=2, step=1)
    slider_grs = st.slider('Jumlah Garasi', min_value=int(df['grs'].min()), max_value=int(df['grs'].max()), value=2, step=1)

    # Load model
    model = joblib.load('model_prediksi_harga_rumah.pkl')

    # Predict house price
    predicted_price = model.predict([[slider_lb, slider_lt, slider_kt, slider_km, slider_grs]])

    if st.button('Prediksi Harga Rumah'):
        formatted_price = f"{predicted_price[0]:,.0f}".replace(',', '.')
        st.write(f"Harga rumah impian anda diperkirakan sekitar IDR {formatted_price} juta")

# Main application logic
def main():
    page = show_sidebar()

    if page == 'Home':
        show_home()
    elif page == 'Prediksi Harga Rumah':
        show_predict()
    elif page == 'Analisis Data':
        show_data()
    elif page == 'Evaluasi Model':
        train_and_evaluate_model()
    elif page == 'Insight Model':
        show_price_distribution()
        show_correlation()

if __name__ == "__main__":
    main()
