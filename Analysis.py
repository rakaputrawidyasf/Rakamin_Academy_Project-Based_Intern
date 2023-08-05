# %% [markdown]
# # Rakamin x ID/X Partners VIX Final Task - Credit Risk Prediction
# 
# *This is the source code for final task of building an end-to-end solution for Credit Risk Prediction*<br>
# Ini adalah source-code untuk tugas membangun solusi end-to-end untuk Credit Risk Prediction
# 
# ## Overview
# Diperlukan sebuah sistem yang mampu memprediksi credit risk, dataset yang digunakan dalam membangun sistem model adalah data pinjaman sebelumnya yang diterima atau ditolak dengan rentang tahun 2007 sampai tahun 2014.
# 
# **Business Metrics:** Loss, Net Profit Margin

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ## Data Preparation
# ### 1) Load data from csv file

# %%
raw_loan_data = pd.read_csv("loan_data_2007_2014.csv", low_memory=False)

# %%
raw_loan_data.head()

# %%
raw_loan_data.info()

# %% [markdown]
# ### 2. Drop empty columns from dataset
# Bila dalam satu kolom tidak ada value/datanya sama sekali, atau null/NaN, maka kolom tersebut dapat diabaikan

# %%
nnc_loan_data = raw_loan_data.dropna(axis=1, how="all")

# %%
nnc_loan_data.info()

# %% [markdown]
# ### 3. Check Loan Status at "loan_status" column
# Status pinjaman terakhir/saat ini pada kolom "loan_status" dipilih sebagai data target untuk prediksi sementara apakah pinjaman yang bersangkutan berisiko atau tidak

# %%
nnc_loan_data["loan_status"].value_counts()

# %% [markdown]
# Karena target prediksi credit risk yang ditentukan hanya dua, yaitu "berisiko" (risky loans) dan "tidak berisiko" (good loans), maka perlu dipastikan nilai dari status akhir pinjaman (loan_status) tidak ambigu. Berdasarkan value yang ada, nilai "Current" dan "In Grace Period" adalah nilai ambigu, tidak menjelaskan apakah status pinjaman selesai/dibayar atau belum/dibebankan. Maka di sini, dikelompokkan nilai/value yang menunjukkan apakah pinjaman bisa dinilai berisiko (risky) atau tidak (good) sebagai berikut:
# 
# 1) **Risky loans:** "Charged Off", "Late (days/periods)", "Does not meet the credit policy. Status:Charged Off", "Default"
# 2) **Good loans:** "Fully Paid", "Does not meet the credit policy. Status:Fully Paid"

# %%
ambiguous = ["Current", "In Grace Period"]
good_loan = ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"]

nnc_loan_data = nnc_loan_data[nnc_loan_data["loan_status"].isin(ambiguous) == False]

# %% [markdown]
# Pemeriksaan kolom "loan_status" untuk memastikan sudah bersih dari nilai/value ambigu

# %%
nnc_loan_data["loan_status"].value_counts()

# %% [markdown]
# ### 4. Visualization
# Visualisasi data hasil pembersihan sementara, untuk mengetahui jumlah/banyaknya loan/pinjaman yang diperkirakan berisiko (risky) maupun tidak (good). Dibuat kolom sementara untuk nilai/value "good" atau "risky", dengan nama kolom "loan_risk_est"

# %%
nnc_loan_data["loan_risk_est"] = np.where(nnc_loan_data["loan_status"].isin(good_loan), "good", "risky")

# %%
plt.figure(figsize=(7, 5))
plt.title("Credit Risk Estimation Visualization")
plt.bar(nnc_loan_data["loan_risk_est"].unique(),
        nnc_loan_data["loan_risk_est"].value_counts())
plt.xlabel("Credit Risk Status")
plt.ylabel("Numbers (est.)")
plt.show()

# %% [markdown]
# ## Exploratory Data Analysis
# ### 1. Column Understanding

# %%
nnc_loan_data.info()

# %% [markdown]
# Berdasarkan pemahaman terhadap kamus data yang tersedia/diberikan, kolom/field dari dataset dikelompokkan sebagai berikut:
# 
# *1) Field yang menjelaskan karakteristik pinjaman/loan*<br>
# "id", "loan_amnt", "funded_amnt", "funded_amnt_inv", "term", "int_rate", "installment", "grade", "sub_grade", "application_type"
# 
# *2) Field yang menjelaskan identitas peminjam/borrower*<br>
# "member_id", "emp_title", "emp_length", "home_ownership", "annual_inc", "verification_status", "zip_code", "addr_state", "dti"
# 
# *3) Field yang menjelaskan rekam jejak peminjam/borrower's personal records*<br>
# "delinq_2yrs", "earliest_cr_line", "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "initial_list_status", "mths_since_last_major_derog", "acc_now_delinq"
# 
# *4) Field yang menjelaskan status akhir pinjaman yang berjalan/loan current status after issued*<br>
# "issue_d", "loan_status", "pymnt_plan", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d"

# %%
loan_chars = ["id",
               "loan_amnt",
               "funded_amnt", "funded_amnt_inv",
               "term",
               "int_rate",
               "installment",
               "grade", "sub_grade",
               "application_type"]

nnc_loan_data[loan_chars].head()

# %%
borrowers = ["member_id",
             "emp_title", "emp_length",
             "home_ownership",
             "annual_inc",
             "verification_status",
             "zip_code", "addr_state",
             "dti"]

nnc_loan_data[borrowers].head()

# %%
borrower_trs = ["delinq_2yrs",
                "earliest_cr_line",
                "inq_last_6mths",
                "mths_since_last_delinq", "mths_since_last_record",
                "open_acc",
                "pub_rec",
                "revol_bal", "revol_util",
                "total_acc",
                "initial_list_status",
                "mths_since_last_major_derog",
                "acc_now_delinq"]

nnc_loan_data[borrower_trs].head()

# %%
loan_stats = ["issue_d",
              "loan_status",
              "pymnt_plan",
              "out_prncp", "out_prncp_inv",
              "total_pymnt", "total_pymnt_inv",
              "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
              "recoveries",
              "collection_recovery_fee",
              "last_pymnt_d", "last_pymnt_amnt",
              "next_pymnt_d"]

nnc_loan_data[loan_stats].head()

# %% [markdown]
# ### 2. Cleaning/Drop Unrequired Columns
# Kolom/field yang menjelaskan status akhir pinjaman atau current loan status, seperti "issue_d", "loan_status", "pymnt_plan", dsb, hanya akan berisi data apabila pinjaman telah ditetapkan (issued). Hanya dengan satu atau sekumpulan kolom ini saja sudah dapat ditentukan mana pinjaman berisiko (risky loans) dan mana yang tidak (good loans), misalnya dari data/kolom "loan_status" sebagaimana yang telah ditunjukkan sebelumnya. Contoh lainnya adalah "out_prncp" (Outstanding Principal), jika bernilai 0 artinya pinjaman telah lunas (Fully Paid), ini sudah cukup menunjukkan bahwa pinjaman tidak berisiko. Kumpulan kolom/fiels ini menjadikan prediksi tidak diperlukan, sehingga kolom/fields ini akan dihapus dari dataset.

# %%
nnc_loan_data = nnc_loan_data.drop(columns=loan_stats, axis=1)

# %% [markdown]
# Kolom selanjutnya yang tidak diperlukan adalah "Unnamed: 0", "id", "member_id", "url", "desc", "zip_code", "title", dan "emp_title". "Unnamed: 0" hanya nomor index baris/record, "id" dan "member_id" adalah nomor identitas, "desc" hanya berisi keterangan/informasi tentang pinjaman yang bersangkutan, "zip_code" adalah kode pos alamat peminjam, "title" berisi judul pinjaman,"emp_title" berisi judul pekerjaan, kolom/fields ini tidak berpengaruh terhadap credit risk.

# %%
unused_cols = ["Unnamed: 0", "id", "member_id", "url", "desc", "zip_code", "title", "emp_title"]

nnc_loan_data = nnc_loan_data.drop(columns=unused_cols, axis=1)

# %% [markdown]
# Selanjutnya adalah memeriksa apakah ada kolom duplikasi, atau kumpulan kolom yang berisi nilai yang sama atau identik agar cukup digunakan salah satunya saja. Pemeriksaan dilakukan pada kolom-kolom numerik.

# %%
nnc_loan_data.select_dtypes(exclude="object")

# %% [markdown]
# Kolom "loan_amnt", "funded_amnt", "funded_amnt_inv" menunjukkan nilai-nilai yang hampir sama, cukup gunakan salah satu saja.<br>
# Dalam notebook ini, kolom yang dipilih adalah "loan_amnt".

# %%
duplicate_cols = ["funded_amnt", "funded_amnt_inv"]

nnc_loan_data = nnc_loan_data.drop(columns=duplicate_cols, axis=1)

# %% [markdown]
# Selanjutnya, dilakukan pemeriksaan data lagi, karena data belum sepenuhnya bersih. Terdapat beberapa hal seperti nilai kosong (NaN), duplikasi, kesalahan format, dan juga outlier.

# %%
nnc_loan_data.info()

# %% [markdown]
# Bila diperhatikan, ada beberapa kolom yang memiliki banyak nilai kosong (NaN), misalnya kolom "mths_since_last_record" (Jumlah bulan sejak catatan publik terakhir), hanya 28949 baris saja yang berisi nilai dari total 238913 baris data. Selain itu, kolom "mths_since_last_major_derog" juga memiliki banyak nilai kosong, hanya 42544 baris saja yang berisi nilai, begitu juga dengan kolom-kolom lain yang banyaknya nilai terisi kurang dari 238913
# 
# Pada bagian ini, dilakukan peninjauan terhadap kolom "mths_since_last_delinq", "mths_since_last_record", dan "mths_since_last_major_derog" karena ketiga kolom ini memiliki nilai kosong terbanyak diantara semua kolom yang ada.

# %%
borrower_pr = ["mths_since_last_delinq", "mths_since_last_record", "mths_since_last_major_derog", "loan_risk_est"]

nnc_loan_data[borrower_pr]

# %% [markdown]
# Bila diperhatikan lagi, sebagian besar record/data dengan ketiga kolom berisi nilai kosong diperkirakan menunjukkan credit risk yang rendah (good). Ditunjukkan pula bahwa semakin kecil nilai/angka, atau jumlah bulan sejak kesalahan atau catatan publik, maka semakin menunjukkan credit risk tinggi (risky).
# 
# Meninjau kembali kamus data, diketahui bahwa kolom "mths_since_last_delinq" berhubungan (related) dengan kolom "delinq_2yrs" dan "acc_now_delinq", dan kolom "mths_since_last_record" berhubungan dengan kolom "pub_rec", sehingga kolom "mths_since_last_delinq" dan "mths_since_last_record" dapat diabaikan dan dihapus (drop). Adapun untuk kolom "mths_since_last_major_derog" dapat diganti dengan kolom baru "major_derogatory" dengan nilai berisi 0 jika nilai "mths_since_last_major_derog" kosong, dan 1 jika berisi nilai.

# %%
unused_cols = ["mths_since_last_delinq", "mths_since_last_record", "mths_since_last_major_derog"]

nnc_loan_data["major_derogatory"] = np.where(nnc_loan_data["mths_since_last_major_derog"].isna(), 0, 1)
nnc_loan_data["major_derogatory"] = nnc_loan_data["major_derogatory"].astype("int64")
nnc_loan_data = nnc_loan_data.drop(unused_cols, axis=1)

# %% [markdown]
# Selanjutnya fokus pada kolom "tot_coll_amt", "tot_cur_bal", dan "total_rev_hi_lim". Ketiganya memiliki jumlah nilai kosong yang sama, tetapi perlu dilakukan pemeriksaan terlebih dahulu, sebab deskripsi ketiga kolom pada kamus data agak membingungkan

# %%
pivot_cols = ["tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim"]

pd.pivot_table(nnc_loan_data, index="loan_risk_est", values=pivot_cols)

# %%
pd.pivot_table(nnc_loan_data, index="loan_risk_est", values=pivot_cols, aggfunc=np.max)

# %% [markdown]
# Berdasarkan tabel pivot dengan nilai max, kolom "tot_coll_amt" terlihat mencurigakan.

# %%
nnc_loan_data[pivot_cols].describe()

# %%
sns.kdeplot(nnc_loan_data[(nnc_loan_data["tot_coll_amt"] < 100000)
                          & nnc_loan_data["tot_coll_amt"] > 0],
            x="tot_coll_amt",
            hue="loan_risk_est")

# %%
sns.kdeplot(nnc_loan_data[nnc_loan_data["tot_cur_bal"] < 800000],
            x="tot_cur_bal",
            hue="loan_risk_est")

# %%
sns.kdeplot(nnc_loan_data[nnc_loan_data["total_rev_hi_lim"] < 250000],
            x="total_rev_hi_lim",
            hue="loan_risk_est")

# %% [markdown]
# Berdasarkan peninjauan dan visualisasi distribusi data kolom "tot_coll_amt", "tot_cur_bal", dan "total_rev_hi_lim", diperoleh kesimpulan berikut:
# * Deskripsi kolom pada kamus data kurang jelas
# * Sekitar 75% data "tot_coll_amt" bernilai 0
# * Hasil visualisasi distribusi data ketiga kolom tidak menunjukkan pemisahan yang jelas antara pinjaman bagus (good) dengan pinjaman berisiko (risky)
# * Bila ketiga kolom tetap atau dipaksakan digunakan dalam analisis data, maka harus menghilangkan lebih dari 50% baris/record
# 
# Untuk meminimalkan kekurangan data dan kesalahan dalam pengembangan model, maka ketiga kolom ini diabaikan/dihapus

# %%
unused_cols = ["tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim"]

nnc_loan_data = nnc_loan_data.drop(unused_cols, axis=1)

# %% [markdown]
# Selanjutnya, dilakukan pemeriksaan untuk mengetahui keragaman nilai data (unique value) pada dataset, untuk mengetahui apakah nilai-nilai data yang berbeda menentukan berisiko atau tidaknya pinjaman

# %%
nnc_loan_data.nunique()[nnc_loan_data.nunique() < 10].sort_values()

# %% [markdown]
# Untuk kolom "policy_code" dan "application_type" hanya memiliki nilai seragam, sehingga dapat diabaikan/dihapus

# %%
unused_cols = ["policy_code", "application_type"]

nnc_loan_data = nnc_loan_data.drop(unused_cols, axis=1)

# %% [markdown]
# ### 3. Categorizing and Imputation
# Selanjutnya dilakukan visualisasi terhadap kolom-kolom dengan keragaman nilai kecil tersebut, untuk mengetahui hubungan antara nilai kolom dengan berisiko atau tidaknya pinjaman (good/risky)

# %%
def field_risk_vis(field):
    ratio = (nnc_loan_data.groupby(field)["loan_risk_est"]
             .value_counts(normalize=True)
             .mul(100)
             .rename("Risky Percentage")
             .reset_index())
    
    sns.lineplot(ratio[ratio["loan_risk_est"] == "risky"],
                 x=field, y="Risky Percentage")
    
    plt.title(field)
    plt.show()

# %%
fields = nnc_loan_data.nunique()[nnc_loan_data.nunique() < 10].sort_values().index
fields = fields.drop("loan_risk_est")

for field in fields:
    field_risk_vis(field)

# %% [markdown]
# Berdasarkan visualisasi di atas, diketahui beberapa kolom memiliki nilai-nilai yang sangat memengaruhi berisiko tidaknya pinjaman, dan sebagian lagi tidak terlalu berpengaruh.
# 
# * Kolom dengan pengaruh besar meliputi "grade", "term", dan "acc_now_delinq"
# * Kolom dengan pengaruh kecil meliputi "initial_list_status", "major_derogatory", "home_ownership", dan "verification_status"
# 
# Selanjutnya, dilakukan pengelompokan antara data kategorikal dan data numerik, lalu dilanjutkan pembersihan data

# %%
# Numeric
num_loan_data = nnc_loan_data.select_dtypes(exclude="object")
num_loan_data.head()

# %%
# Categorical
cat_loan_data = nnc_loan_data.select_dtypes(include="object")
cat_loan_data

# %% [markdown]
# Bila diperhatikan baik-baik pada bagian kolom-kolom kategorikal, kolom "emp_length" seharusnya bertipe numerik, kolom "earliest_cr_line" dan "last_credit_pull_d" seharusnya bertipe datetime.
# 
# Bila membandingkan ketiga kolom dengan nilai good/risky pada kolom "loan_rsik_est", serta merujuk pada kamus data, maka diasumsikan:
# * Lama pekerjaan (kolom "emp_length") bernilai antara 0 sampai 10 tahun, dimana 0 artinya kurang dari setahun dan 10 artinya 10 tahun atau lebih. Semakin besar nilai (0 ke 10) pada kolom, maka semakin kecil risiko pinjaman (ditunjukkan dengan nilai "good" pada kolom "loan_risk_est")
# * Bulan jalur kredit paling awal (kolom "earliest_cr_line"), semakin lama/kebelakang tahun umumnya semakin baik rekam jejaknya
# * Kolom "last_credit_pull_d" atau penarikan kredit terakhir, semakin lama jeda waktu antara pertanyaan (inquiry) dengan waktu terakhir (today) maka semakin baik
# 
# Akan tetapi, perlu dipastikan dulu apakah asumsi tersebut sudah benar. Untuk memudahkan, dilakukan pengkodean nilai kategorikal menjadi nilai numerik.

# %%
nnc_loan_data["emp_length"].unique()

# %%
emp_length = ["< 1 year", "1 years", "2 years", "3 years", "4 years",
              "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"]

emp_map = dict()
i = 0

for item in emp_length:
    emp_map[item] = i
    i = i + 1

emp_map

# %%
nnc_loan_data["emp_length"] = cat_loan_data["emp_length"].map(emp_map).fillna("0").astype("int64")
nnc_loan_data[["emp_length", "loan_risk_est"]]

# %% [markdown]
# Berikut adalah transformasi data kolom "earliest_cr_line" dan "last_credit_pull_d" dari kategorikal menjadi datetime (numerik)

# %%
cat_loan_data["earliest_cr_year"] = pd.to_datetime(
    cat_loan_data["earliest_cr_line"],
    format="%b-%y"
).dt.year

nnc_loan_data["earliest_cr_line"] = np.where(
    cat_loan_data["earliest_cr_year"] > 2023,
    cat_loan_data["earliest_cr_year"] - 100,
    cat_loan_data["earliest_cr_year"]
)

nnc_loan_data["yr_since_last_inq"] = 2016 - pd.to_datetime(
    cat_loan_data["last_credit_pull_d"],
    format="%b-%y"
).dt.year

nnc_loan_data = nnc_loan_data.drop("last_credit_pull_d", axis=1)

nnc_loan_data[
    ["emp_length", "earliest_cr_line", "yr_since_last_inq"]
].describe()

# %% [markdown]
# Bila diperhatikan lagi, kolom "earliest_cr_line" dan "yr_since_last_inq" memiliki nilai kosong. Kolom "earliest_cr_line" hanya terisi 238884 nilai, dan kolom "yr_since_last_inq" hanya terisi 238890 dari total 238913.
# 
# Selanjutnya, dilakukan peninjauan lagi terhadap dataset untuk memastikan banyaknya nilai kosong pada masing-masing kolom. Disini dipisahkan antara kolom/data numerik dan kolom/data kategorikal

# %%
# Kolom numerik
num_loan_data = nnc_loan_data.select_dtypes(exclude="object")

num_loan_data.isna().sum()

# %%
# Kolom kategorikal
cat_loan_data = nnc_loan_data.select_dtypes(include="object")

cat_loan_data.isna().sum()

# %% [markdown]
# Terlihat bahwa sebagian kolom numerik memiliki nilai-nilai kosong, sementara kolom kategorikal semuanya terisi dan tidak ada nilai kosong. Karena nilai kosong bertipe numerik, maka dilakukan pengisian nilai kosong dengan imputasi multivariat menggunakan library "sklearn.impute.IterativeImputer"

# %%
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

nan_cols = ["annual_inc", "delinq_2yrs", "earliest_cr_line",
            "inq_last_6mths", "open_acc", "pub_rec", "revol_util",
            "total_acc", "collections_12_mths_ex_med",
            "acc_now_delinq", "yr_since_last_inq"]

imputer = IterativeImputer(max_iter=10, random_state=0)
num_loan_data[nan_cols] = imputer.fit_transform(num_loan_data[nan_cols])

# %%
num_loan_data.isna().sum()

# %% [markdown]
# ### 5. Visualization
# Selanjutnya, kembali ke kolom "emp_length", "earliest_cr_line", dan "yr_since_last_inq" untuk dilakukan visualisasi, untuk mengetahui hubungan antara nilai kolom dengan berisiko atau tidaknya pinjaman (good/risky)

# %%
fields = ["emp_length", "earliest_cr_line", "yr_since_last_inq"]

for field in fields:
    field_risk_vis(field)

# %% [markdown]
# Berdasarkan visualisasi ketiga kolom tersebut, diperoleh kesimpulan sementara sebagai berikut:
# * Lama pekerjaan ("emp_length") memiliki keragaman nilai, nilai kurang dari 1 tahun menunjukkan credit risk paling besar
# * Bulan jalur kredit paling awal ("earliest_cr_line") sebelum 1960 menunjukkan credit risk paling besar
# * Kolom "yr_since_last_inq" (Lama tahun sejak pertanyaan terakhir) menunjukkan credit risk paling besar antara nilai 6 dan 8
# 
# Selanjutnya, dilakukan visualisasi terhadap data numerik dan kategorikal untuk pemeriksaan berikut:
# 
# **1) Numerik**
# * Histogram (distribusi data)
# * Matriks Korelasi
# * Pivot Table
# 
# **2) Kategorikal**
# * Balance (keseimbangan data)
# * Pivot Table
# 
# **Numerik:**

# %%
for column in num_loan_data.columns:
    plt.hist(num_loan_data[column])
    plt.title(column)
    plt.show()

# %%
plt.figure(figsize=(12, 6))
sns.heatmap(data=num_loan_data.corr(), annot=True)

# %%
pd.pivot_table(nnc_loan_data,
               index="loan_risk_est",
               values=num_loan_data.columns)

# %% [markdown]
# **Kesimpulan:**
# * Sebagian besar data numerik tidak terdistribusi secara normal
# * Beberapa data memiliki outlier
# * Diketahui bahwa kolom "installment" (Pembayaran bulanan yang terhutang oleh peminjam) berkorelasi dengan "loan_amnt" (Bulan lalu pembayaran diterima) sebesar 0,96 (96%), karena "installment" = "loan_amnt" * "int_rate" (Indikator pedapatan diverifikasi, tidak diferivikasi, atau sumber pendapatan diverifikasi)
# 
# Berdasarkan pivot table, kemungkinan suatu pinjaman berisiko (risky) ditunjukkan dengan indikasi berikut:
# * Berdasarkan rekam jejak personal:
#     - nilai "acc_now_delinq" (Jumlah akun di mana peminjam sekarang nakal) lebih tinggi
#     - nilai "delinq_2yrs" (Jumlah 30+ hari insiden kenakalan dalam file kredit peminjam selama 2 tahun terakhir) lebih tinggi
#     - nilai "inq_last_6months" (Jumlah pertanyaan dalam 6 bulan terakhir) lebih tinggi
#     - nilai "yr_since_last_inq" (Lama tahun sejak pertanyaan terakhir) lebih rendah
# <br><br>
# * Berdasarkan tingkat kesulitan pembayaran:
#     - nilai "annual_inc" lebih rendah
#     - nilai "dti" (Rasio dihitung menggunakan total pembayaran bulanan peminjam bersama atas total kewajiban utang) lebih tinggi
#     - nilai "installment" dan "loan_amnt" lebih tinggi
#     - nilai "int_rate" lebih tinggi
# <br><br>
# * Nilai "collections_12_mths_ex_med" (Jumlah koleksi dalam 12 bulan tidak termasuk koleksi medis) lebih tinggi
# * Nilai "revol_util" (Tingkat pemanfaatan jalur revolving, atau jumlah kredit peminjam menggunakan relatif terhadap semua kredit revolving yang tersedia) lebih tinggi
# 
# **Kategorikal:**

# %%
cat_loan_data.nunique()

# %%
fields = ["grade", "sub_grade", "home_ownership",
          "verification_status", "purpose", "addr_state",
          "initial_list_status"]

for field in fields:
    plt.figure(figsize=(12, 4))
    field_risk_vis(field)

# %% [markdown]
# **Kesimpulan:**<br>
# * Pada kolom "grade" (LC menugaskan nilai pinjaman) dan "sub_grade" (LC Ditugaskan Subgrade Pinjaman) menunjukkan bahwa semakin tinggi (ordinal) nilai pada kedua kolom, maka semakin tinggi risiko pinjaman
# * Pada kolom "home_ownership" (Status kepemilikan rumah yang disediakan oleh peminjam selama pendaftaran), nilai "NONE" menunjukkan risiko rendah (good), sementara nilai "RENT" menunjukkan risiko tinggi (risky)
# * Pada kolom "verification_status" (Indikator jika pendapatan bersama co-peminjam diverifikasi oleh LC, tidak diverifikasi, atau jika sumber pendapatan diverifikasi), nilai "Not Verified" mengindikasikan risiko rendah (good)
# * Pada kolom "purpose" (), nilai "car" dan "wedding" menunjukkan risiko rendah (good), sementara nilai "small_business" menunjukkan risiko tinggi (risky)
# * Pada kolom "initial_list_status" (), nilai "f" menunjukkan risiko rendah (good) sementara nilai "w" menunjukkan risiko tinggi (risky)
# 
# ### 6. Data Transformation
# Selanjutnya, dilakukan perubahan/transformasi kolom-kolom kategorikal menjadi kolom-kolom numerik untuk memudahkan pengembangan. Sebelumnya, dilakukan penghapusan kolom "sub_grade", karena kolom ini berhubungan dengan kolom "grade".

# %%
cat_loan_data = cat_loan_data.drop("sub_grade", axis=1)

cat_loan_data.nunique()

# %% [markdown]
# * Pada kolom "terms", cukup menghilangkan kata/string " months", dan secara otomatis menjadi kolom numerik
# * Pada kolom "initial_list_status", nilai diubah menjadi boolean dengan nilai "f" menjadi "0" dan "w" menjadi "1"
# * Untuk kolom "grade" dan "addr_state", dilakukan ordinal encoding/pengkodean seperti pada kolom "emp_length"
# * Untuk kolom "home_ownership", "verification_status", dan "purpose", dilakukan One Hot Encoding

# %%
# Untuk kolom "terms"
cat_loan_data["term"] = cat_loan_data["term"].str.replace(" months", "").astype("int64")

# %%
# Untuk kolom "initial_list_status"
cat_loan_data["initial_ls_stats_w"] = np.where(cat_loan_data["initial_list_status"] == "w", 1, 0).astype(int)
cat_loan_data = cat_loan_data.drop("initial_list_status", axis=1)
cat_loan_data["initial_list_status"] = cat_loan_data["initial_ls_stats_w"].astype("int64")
cat_loan_data = cat_loan_data.drop("initial_ls_stats_w", axis=1)

# %%
# Untuk kolom "grade"
grade = cat_loan_data["grade"].unique()
grade.sort()

grade_map = dict()
i = 1

for item in grade:
    grade_map[item] = i
    i = i + 1

grade_map

# %%
# Untuk kolom "addr_state"
addr_state = cat_loan_data["addr_state"].unique()
addr_state.sort()

addr_st_map = dict()
i = 1

for item in addr_state:
    addr_st_map[item] = i
    i = i + 1

addr_st_map

# %%
cat_loan_data["grade"] = cat_loan_data["grade"].map(grade_map).astype("int64")
cat_loan_data["addr_state"] = cat_loan_data["addr_state"].map(addr_st_map).astype("int64")

cat_loan_data.head()

# %%
# Untuk kolom "home_ownership", "verification_status", dan "purpose"
fields = ["home_ownership", "verification_status", "purpose"]

dummy_loan_data = pd.get_dummies(cat_loan_data[fields])
dummy_loan_data.head()

# %% [markdown]
# ### 7. Concatenation
# Setelah dilakukan pemisahan, pembersihan, dan perubahan pada masing-masing kolom kategorikal dan numerik, selanjutnya adalah menggabungkan kolom-kolom tersebut menjadi satu dataset yang siap untuk dianalisis

# %%
final_cat_ld = cat_loan_data.drop(fields, axis=1)
final_cat_ld = pd.concat([final_cat_ld, dummy_loan_data], axis=1)
final_cat_ld.head()

# %%
loan_risk_est = final_cat_ld["loan_risk_est"]
final_loan_data = pd.concat(
    [final_cat_ld.drop("loan_risk_est", axis=1), num_loan_data],
    axis=1
)

final_loan_data.head()

# %%
final_loan_data.info()

# %% [markdown]
# ## Modelling
# Setelah persiapan dan pemersihan dataset selesai, diperoleh dataset yang siap untuk dianalisis. Selanjutnya, dilakukan pengembangan model Machine Learning untuk melakukan analisis pada dataset.
# 
# ### 1. Preprocessing
# Berikut adalah pembagian dataset untuk pelatihan (training) dan validasi (validation). Untuk memudahkan pengembangan, nilai prediksi credit risk (kolom "loan_risk_est") yang masih kategorikal diubah menjadi numerik, nilai "good" diubah menjadi nilai 0 dan nilai "risky" menjadi nilai 1

# %%
from sklearn.model_selection import train_test_split

X = final_loan_data
y = np.where(loan_risk_est == "risky", 1, 0)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# %% [markdown]
# ### 2. Modelling and Evaluation
# Selanjutnya adalah pemilihan algoritma untuk membangun model untuk analisis dataset. Algoritma yang akan digunakan dalam pemodelan analisis/klasifikasi disini adalah:
# 
# * Naive Bayes (Gaussian)
# * K-Nearest Neighbors
# * Decision Tree
# * Random Forest
# 
# Setelah dilakukan pemodelan, selanjutnya adalah mengevaluasi model yang telah dikembangkan. Komponen evaluasi model meliputi nilai-nilai berikut:
# 
# * *Precision*: Adalah rasio banyaknya data yang teranggap sebagai anggota suatu kelas/label dari keseluruhan data yang diprediksi sebagai anggota kelas/label tersebut. Dalam hal ini, banyaknya data pinjaman yang teranggap berisiko (risky) dari kumpulan data pinjaman yang diprediksi berisiko
# 
# * *Recall*: Adalah rasio banyaknya data yang terprediksi sebagai anggota suatu kelas/label dari keseluruhan data yang sebenarnya termasuk anggota kelas/label tersebut. Dalam hal ini, banyaknya data pinjaman yang terprediksi berisiko (risky) dari kumpulan data pinjaman yang sebetulnya berisiko

# %%
# For Modelling
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# For Evaluation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# %% [markdown]
# Keterangan kelas/label:
# * **0**: "good" (pinjaman tidak berisiko)
# * **1**: "risky" (pinjaman berisiko)

# %%
# GAussian Naive Bayes
gnb = GaussianNB()
gnb.fit(train_X, train_y)
predict_y = gnb.predict(val_X)
print(classification_report(val_y, predict_y))

# %%
# K-Nearest Neighbors (n=5)
knn = KNeighborsClassifier()
knn.fit(train_X, train_y)
predict_y = knn.predict(val_X)
print(classification_report(val_y, predict_y))

# %%
# Decision Tree
dtree = DecisionTreeClassifier(random_state=10)
dtree.fit(train_X, train_y)
predict_y = dtree.predict(val_X)
print(classification_report(val_y, predict_y))

# %%
# Random Forest
rfc = RandomForestClassifier(random_state=10)
rfc.fit(train_X, train_y)
predict_y = rfc.predict(val_X)
print(classification_report(val_y, predict_y))

# %% [markdown]
# ## Conclusion
# Dari proses persiapan data, pembersihan, transformasi data, sampai tahap pemodelan dan evaluasi, diperoleh kesimpulan:
# 
# * Dataset asli memiliki beberapa kolom yang kosong/tidak ada nilai samasekali, sehingga kolom-kolom yang kosong lebih baik diabaikan/dihapus
# * Sebagian kolom berisi identitas peminjam serta status akhir pinjaman. Identitas tidak berpengaruh terhadap prediksi credit risk, sedangkan status akhir pinjaman dapat langsung menunjukkan apakah pinjaman tersebut berisiko atau tidak, sehingga tidak perlu pengembangan model dan prediksi. Karena itu, kolom-kolom terkait identitas dan status akhir pinjaman diabaikan/dihapus
# * Perlunya pemisahan sementara antara kolom kategorikal dengan kolom numerik, karena kedua kolom memerlukan penanganan yang berbeda
# * Sebagian kolom memiliki banyak nilai kosong. Jika dipaksakan untuk menggunakan semuanya, maka harus menghilangkan lebih dari 50% baris data. Adapun bila nilai kosongnya sedikit, maka masih bisa diatasi dengan imputasi data
# * Terkadang beberapa kolom memiliki format yang salah, misal kolom kategorikal yang seharusnya numerik karena menunjukkan jangka waktu
# * Setelah pembersihan kedua jenis data, perlu dilakukan transformasi pada data kategorikal menjadi numerik untuk memudahkan pengembangan model
# * Setelah semua data ditransformasi dan digabungkan menjadi satu dataset siap pakai, dilanjutkan ke tahap preprocessing
# * Preprocessing data meliputi pemisahan data dengan kelas/label, dalam hal ini adalah perkiraan credit risk berisiko atau tidak, kemudian dilakukan pemisahan antara data latih (training) dan data uji/validasi (test/validation)
# * Selanjutnya, dilakukan pemodelan untuk pelatihan data latih dan prediksi data uji, menggunakan algoritma/model yang tepat sesuai kebutuhan
# * Evaluasi model dilakukan untuk meninjau apakah model sudah mampu memprediksi data dengan baik atau belum

# %%
"""Sekian dan Terimakasih"""


