import streamlit as st 
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import altair as alt
from datetime import timedelta, date, datetime
from wordcloud import WordCloud
from streamlit_option_menu import option_menu
import keras
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from stqdm import stqdm
from google_play_scraper import Sort, reviews
from streamlit_gsheets import GSheetsConnection
import pytz

# import streamlit_wordcloud as wordcloud
from numerize.numerize import numerize
import json

# page config
st.set_page_config(
    page_icon="💳",
    page_title="Sentiment BSI Mobile", 
    layout = "wide"
)
# data preparation 
logo = Image.open("image/bsi.png")

def load_data(id, sheet_name)-> pd.DataFrame:
  url = f'https://docs.google.com/spreadsheets/d/{id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
  return pd.read_csv(url, parse_dates=[0, 1, 2, 3, 4], infer_datetime_format=True)

id = "1oTSMV4_VAoZEvU7-bHJLoKsg-mSGpFhxyFQLSOjl3VI"
sheet_name = "temp"

df1_tambahan = load_data(id, sheet_name)
df1_tambahan["datetime_baru"]= pd.to_datetime(df1_tambahan["datetime_baru"])
df1_tambahan["score"] = df1_tambahan["score"].astype(int)
df1_tambahan["prob_keyakinan"] = df1_tambahan["prob_keyakinan"].astype(float)
maksDate = max(df1_tambahan["datetime_baru"].dt.date)

# Use the following line to include your style.css file
# st.markdown('<style>' + open('style.css').read() + '</style>', unsafe_allow_html=True)
def social_icons(width=24, height=24, **kwargs):
        icon_template = '''
        <a href="{url}" target="_blank" style="margin-right: 20px;">
            <img src="{icon_src}" alt="{alt_text}" width="{width}" height="{height}">
        </a>
        '''

        icons_html = ""
        for name, url in kwargs.items():
            icon_src = {
                "linkedin": "https://img.icons8.com/ios-filled/100/ff8c00/linkedin.png",
                "github": "https://img.icons8.com/ios-filled/100/ff8c00/github--v2.png",
                "email": "https://img.icons8.com/ios-filled/100/ff8c00/filled-message.png"
            }.get(name.lower())

            if icon_src:
                icons_html += icon_template.format(url=url, icon_src=icon_src, alt_text=name.capitalize(), width=width, height=height)

        return icons_html
#####################

### buat prediksi sentiment 
# tokenizer preprocessing
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
max_length = 120
trunc_type='post'
pad_type='post'
oov_tok = "<OOV>"

# emot 
emoticon = pd.read_csv("dataset/master_emoji.csv")
emot_f= emoticon[["Emoji", "tag_indo", "Sentiment"]].copy()
def replace_emot(teks, emot_f):
    for j, emoticon in enumerate(emot_f["Emoji"]):
        tag = emot_f["tag_indo"][j]
        teks = teks.replace(emoticon, f" {tag}")
    return teks

# slang normalize 
slang_dict = pd.read_csv('dataset/new_kamusalay.csv', encoding='latin-1', header=None)
slang_dict = slang_dict.rename(columns={0: 'original',
                                    1: 'replacement'})
def normalize_slang(text):
    slang_dict_map = dict(zip(slang_dict['original'], slang_dict['replacement']))
    return ' '.join([slang_dict_map[word] if word in slang_dict_map else word for word in text.split(' ')])

# stop_words preprocessing 
STOP_PREP = []
with open("stop_preprocessing.txt", "r") as infile:
    STOP_PREP = infile.read().splitlines()

# main preprocess 
def preprocessing(text):
    text = replace_emot(text, emot_f)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'username', '', text)
    text = normalize_slang(text)

    # Mengubah string menjadi list kata
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in STOP_PREP]
    text = ' '.join(filtered_words)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r'[ ]+', ' ', text)

    return text.strip()

# preprocessing 
def teks_to_pad(teks):
    teks_seq = tokenizer.texts_to_sequences(teks)
    teks_pad = pad_sequences(teks_seq, maxlen=max_length, truncating=trunc_type, padding=pad_type)
    return teks_pad

utc_now = datetime.now(pytz.utc)
text1, pict1 = st.columns((4,2))
with text1:
    st.title("Analisis Sentiment Ulasan BSI Mobile [Daily Batching]")
    st.subheader("Also as Sentiment Analysis Tools")
    st.write(f"(Last Data Date: {maksDate})")
    st.write(f"Server Time (UTC+0): {utc_now}")
with pict1:
    st.image(logo)
    # st.write("Ludy Hasby Aulia - nMLE")

# batas atas scrapping eksluksif 
end_date = utc_now.date()
# batas bawah scrapping ekslusif 
start_date = maksDate
diff_days = (end_date - start_date).days
counter_scrapper = diff_days*100

if diff_days >= 2:
    with st.expander("Update Data Dulu ya, sudah waktunya 😁"):
        url = "https://docs.google.com/spreadsheets/d/1oTSMV4_VAoZEvU7-bHJLoKsg-mSGpFhxyFQLSOjl3VI"
        conn = st.connection("gsheets", type=GSheetsConnection)
        conn.read(spreadsheet=url, worksheet="temp", usecols=list(range(7)))

        st.write("Srapping on Progress")
        review, token = reviews(
            'com.bsm.activity2',
            lang='id',
            country='id',
            sort = Sort.NEWEST,
            filter_score_with=None,
            count=counter_scrapper
        )
        review_df = pd.DataFrame(review)
        st.write("Scrapping Done..")
        if not review_df.empty:
            review_df["datetime_baru"] = pd.to_datetime(review_df["at"])
            review_df = review_df[(review_df["datetime_baru"].dt.date >= start_date) & (review_df["datetime_baru"].dt.date < end_date)]
            date_scrap_bellow = min(review_df["datetime_baru"].dt.date)

            if date_scrap_bellow != (start_date):
                st.write("Scrapping 2 on Progress")
                review, token = reviews(
                    'com.bsm.activity2',
                    lang='id',
                    country='id',
                    sort = Sort.NEWEST,
                    filter_score_with=None,
                    count=counter_scrapper*2
                    
                )
                review_df = pd.DataFrame(review)
                st.write("Scrapping 2 Done")
                review_df["datetime_baru"] = pd.to_datetime(review_df["at"])
                review_df = review_df[(review_df["datetime_baru"].dt.date >= start_date) & (review_df["datetime_baru"].dt.date < end_date)]

            model = keras.models.load_model("model/sentiment_5.h5")
            # emot 
            emoticon = pd.read_csv("dataset/master_emoji.csv")
            emot_f= emoticon[["Emoji", "tag_indo", "Sentiment"]].copy()

            # slang normalize 
            slang_dict = pd.read_csv('dataset/new_kamusalay.csv', encoding='latin-1', header=None)
            slang_dict = slang_dict.rename(columns={0: 'original',
                                                1: 'replacement'})
            # stop_words preprocessing 
            STOP_PREP = []
            with open("stop_preprocessing.txt", "r") as infile:
                STOP_PREP = infile.read().splitlines()

            st.write("Preprocessing on Progress..")
            stqdm.pandas()
            review_df["Preprocess_Sentences"] = review_df["content"].progress_apply(preprocessing)
            st.write("Handling Null..")
            review_df = review_df[review_df["Preprocess_Sentences"].notna()]
            review_df["sentences_wordCloud"] = review_df["Preprocess_Sentences"]
            preprocess_sentences = review_df["Preprocess_Sentences"].tolist()
            st.write("Padding on Progress..")
            padding_sentences = teks_to_pad(preprocess_sentences)
            st.write("Prediksi dimulai..")
            result = model.predict(padding_sentences)
            review_df["prob_keyakinan"] = result
            review_df["sentiment"] = ["positive" if i >= 0.5 else "negative" for i in result]
            st.write("All Data Have Been Succeed Processed..")

            review_df = review_df[["content", "score", "sentences_wordCloud", "Preprocess_Sentences", "prob_keyakinan", "sentiment", "datetime_baru"]]
            review_df.columns = ["Sentences", "score", "sentences_wordCloud", "Preprocess_Sentences", "prob_keyakinan", "sentiment", "datetime_baru"]
            review_df["score"] = review_df["score"].astype(int)
            review_df["prob_keyakinan"] = review_df["prob_keyakinan"].astype(float)
            data = pd.concat([df1_tambahan, review_df])
            # karena tadi diambil data sehari sebelum data yang sudah ada di database, akan di drop yang sama (hampir dipastikan ada duplikat)
            data = data.drop_duplicates(keep=False)
            data = data.sort_values("datetime_baru")
            data["datetime_baru"] = data["datetime_baru"].astype(str)
            conn.update(worksheet="temp", data=data)
            st.write("Append To Database Succcess !")
            df1_tambahan = load_data(id, sheet_name)
            df1_tambahan["datetime_baru"]= pd.to_datetime(df1_tambahan["datetime_baru"])
            df1_tambahan["score"] = df1_tambahan["score"].astype(int)
        else:
            st.write("Tidak ada Data Masuk 😅")
 
df11 = pd.read_csv("dataset/ps_processing.csv")
df11["datetime_baru"] = pd.to_datetime(df11["datetime_baru"])
df1 = pd.concat([df11, df1_tambahan])
df1["datetime_baru"] = pd.to_datetime(df1["datetime_baru"])
df1["sumber"] = "Play Store"
df2 = pd.read_csv("dataset/as_processing.csv")
df2["datetime_baru"] = pd.to_datetime(df2["datetime_baru"])
df2["sumber"] = "Apps Store"
df = pd.concat([df1, df2])
# df["tahun_bulan"] = df["datetime_baru"].dt.to_period('M')
df["prob_keyakinan"] = df["prob_keyakinan"].astype(float)
df["jam"] = df["datetime_baru"].dt.hour
df["bulan"] = df["datetime_baru"].dt.month
df["date_day"] = df["datetime_baru"].dt.day
df["dayNames"] = df["datetime_baru"].dt.day_name()
df["tanggal"] = df["datetime_baru"].dt.date
bsi_date = date.fromisoformat('2021-02-01')
df = df[df["tanggal"] >= bsi_date]

df["tahun"] = df["datetime_baru"].dt.year
df["tahun_bulan"] = df["tahun"].astype(str) + "-" + df["bulan"].astype(str)
df["tahun_bulan_01"] = df["tahun_bulan"] + "-01"
minDate = min(df["tanggal"])
maksDate = max(df["tanggal"])

with st.sidebar:
     choose = option_menu(
                        "Outline",
                        ["Distribusi Rating BSI Mobile", 
                         "Distribusi Rerata Rating BSI Mobile dalam Periode", 
                         "Perbandingan Hasil Analisis Sentiment dengan Rating BSI Mobile", 
                         "Sentiment antar Periode", 
                         "Sentiment antar tahun",
                         "Sentiment Positive VS Negative Antar Jam", 
                         "Sentiment Positive VS Negative Antar Bulan", 
                         "Sentiment Positive VS Negative Antar Tanggal", 
                         "Sentiment Positive VS Negative Antar Hari", 
                         "Sentiment Berdasar Versi Apps",
                         "WordCloud Generator",
                         "Contoh Ulasan Bersentiment"],
                        icons = ['bi bi-bar-chart-fill', 
                                    'bi bi-bezier2', 
                                    'bi bi-pie-chart', 
                                    'bi bi-textarea-resize', 
                                    'bi bi-stack',
                                    'bi bi-alarm-fill', 
                                    'bi bi-calendar-month', 
                                    'bi bi-calendar-date-fill', 
                                    'bi bi-calendar-day', 
                                    'bi bi-window-split',
                                    'bi bi-cloud-haze-fill',
                                    'bi bi-chat-left-dots'],
                        menu_icon="mortadboard",
                        default_index=0,
                        styles={
        "container": {"padding": "0!important", "background-color": "#839E9E"},
        "icon": {"color": "darkorange", "font-size": "20px"}, 
        "nav-link": {"font-size": "17px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#5B6464"},
    }
    )
     st.write(":red[Note]: Mohon maaf Outline diatas tidak akan over anda ke halaman, hanya sebagai petunjuk ketersediaan fitur pada tab ini.")
     linkedin_url = "https://www.linkedin.com/in/ludy-hasby/"
     github_url = "https://github.com/ludyhasby"
     email_url = "mailto:ludy.hasby@gmail.com"

     with st.container():
        l, m, r = st.columns((0.11,2,0.1))
        with l:
            st.empty()
        with m:
            st.markdown(
                social_icons(30, 30, LinkedIn=linkedin_url, GitHub=github_url, Email=email_url),
                unsafe_allow_html=True)
        with r:
            st.empty()
        
        if choose == "Test":
            js = """
            <script>
            window.location.href = 'http://localhost:8501/#distribusi-rating-bsi-mobile';
            </script>
            """
            st.markdown(js, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Eksplorasi Sentiment Result🌍",
                            "Perbandingan Performa🏦", 
                            "Credits Training - Contact👋"])
with tab1:
    ## Filter 
    st.write("Seleksi Filter, silahkan dipilih range waktu yang diinginkan")
    kolom1, kolom2, kolom3, kolom4 = st.columns(4)
    with kolom1:
        PERIOD = st.selectbox("ShortCut Time", options=["All Times", "2024", "Minggu Terakhir", "Custom Date", "Pilih Bulan"], index=0)
    with kolom2:
        rating_selection = st.multiselect("Rating", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    with kolom3:
        positiveNegative = st.multiselect("Jenis Sentiment", options=["positive", "negative"], default=["positive", "negative"])
    with kolom4: 
        sumber = st.multiselect("Platform", options=["Play Store", "Apps Store"], default=["Play Store", "Apps Store"])
    df = df[(df["score"].isin(rating_selection)) & (df["sentiment"].isin(positiveNegative)) & (df["sumber"].isin(sumber))]
    if PERIOD == "2024":
        df = df[df["datetime_baru"].dt.year==2024]
    elif PERIOD == "Minggu Terakhir":
        BatasMinggu = maksDate - timedelta(days=7)
        df = df[(df["tanggal"]>=BatasMinggu) & (df["tanggal"]<=maksDate)]
        PERIOD = str(BatasMinggu) + " - "+ str(maksDate)
    elif PERIOD == "Custom Date":
        start_date, end_date = st.date_input("For Custome Date, Pick a Date Please", (minDate, maksDate),min_value=minDate, max_value=maksDate)
        df = df[(df["tanggal"]>=start_date) & (df["tanggal"]<=end_date)]
        PERIOD = str(start_date) + " - "+ str(end_date)
    elif PERIOD == "Pilih Bulan":
        bulan = st.multiselect("Pilih Nomor Bulan", options= sorted(df["bulan"].unique()), default=sorted(df["bulan"].unique()))
        st.caption("Notes: 1 = Januari, 12 = Desember, dst.")
        df = df[df["bulan"].isin(bulan)]
    elif PERIOD == "All Times":
        PERIOD = PERIOD + " (" + str(minDate) + " - "+ str(maksDate) + ")"

    ## Metric 
    positive = df[df["sentiment"]=="positive"]
    negative = df[df["sentiment"]=="negative"]
    kol1, kol2, kol3 = st.columns(3)
    with kol1:
        col1, col2 = st.columns(2)
        with col1: 
            st.metric("Dataset Tersedia", numerize(len(df)), label_visibility="visible")
        with col2:
            play_store_count = len(df[df['sumber'] == 'Play Store'])
            apps_store_count = len(df[df['sumber'] == 'Apps Store'])
            st.write(f"Play Store: :blue[{numerize(play_store_count)}]")
            st.write(f"Apps Store: :blue[{numerize(apps_store_count)}]")
    kol2.metric("Total Sentiment Positif", numerize(len(positive)), label_visibility="visible")
    kol3.metric("Total Sentiment Negative", numerize(len(negative)), label_visibility="visible")


    
        
    ## Score
    st.header(f"Rating BSI Mobile pada :blue[{PERIOD}]")
    # score distribution
    st.subheader("Distribusi Rating BSI Mobile")
    custom_order = [1, 2, 3, 4, 5]
    dist_score = df.score.value_counts().reindex(custom_order)
    dist_score = dist_score.reset_index()
    dist_score.columns = ['rating', 'jumlah']
    dist_score['rating'] = dist_score['rating'].astype(str)

    bar = alt.Chart(dist_score).mark_bar(color="#3EA5A1").encode(
        x = 'rating',
        y = 'jumlah'
    )
    st.altair_chart((bar), use_container_width=True) # legenda rating 

    # score average 
    st.subheader("Distribusi Rerata Rating BSI Mobile dalam Periode")
    coba = df.groupby(['tahun_bulan_01'])['score'].mean().reset_index()
    coba.columns = ["periode", "period_avg_rating"]

    lines = alt.Chart(coba).mark_line(color="#3EA5A1").encode(
        x = alt.X('periode:T', axis=alt.Axis(labelAngle=-45,
            format='%b-%Y'  # Mengatur format label menjadi "bulan-tahun"
        ), title="Periode (Bulan-Tahun)"), 
        y = alt.Y('period_avg_rating:Q', title="Rerata Rating Periodik")
    )

    lines.configure_legend(
        strokeColor='red',
        fillColor='#EEEEEE',
        padding=10,
        orient='bottom-left'
    )
    rerata = np.mean(df["score"])
    avg_total = alt.Chart(pd.DataFrame({'y': [rerata]})).mark_rule(color="red").encode(
    y='y:Q')
    avg_periodic = alt.Chart(coba).mark_rule(color="blue").encode(
        y = alt.Y('mean(period_avg_rating):Q')
    )
    st.altair_chart((lines+avg_total+avg_periodic), use_container_width=True)
    avg_per = np.mean(coba["period_avg_rating"])
    st.write(f"Legenda: :green[rerata rating periodik], :red[rerata rating semua] ({rerata:.2f}), :blue[rerata rating semua periode] ({avg_per:.2f})")

    # rerata harian
    detail_on = st.toggle("Lihat Rerata Harian")

    if (detail_on):
        detail = df.groupby(['tanggal'])['score'].mean().reset_index()
        detail.columns = ["tanggal", "daily_avg_rating"]
        lines = alt.Chart(detail).mark_line(color="#3EA5A1").encode(
            x = 'tanggal',
            y = 'daily_avg_rating:Q'
        ).interactive()

        lines.configure_legend(
            strokeColor='red',
            fillColor='#EEEEEE',
            padding=10,
            orient='bottom-left'
        )

        avg_total = alt.Chart(detail).mark_rule(color="blue").encode(
            y = 'mean(daily_avg_rating):Q'
        )
        st.altair_chart((lines+avg_total), use_container_width=True)


    ## Sentiment
    st.header("Sentiment BSI Mobile Apps")
    sentiment = df.groupby(["sentiment"]).size().reset_index()
    sentiment.columns = ["sentiment", "size"]
    st.markdown(f'<span style="font-size: 22px;">**Perbandingan Hasil** Analisis Sentiment dengan Rating BSI Mobile :blue[{PERIOD}]</span>', unsafe_allow_html=True)
    kol1, kol2 = st.columns(2)
    with kol1:
        fig = px.pie(sentiment, values='size', names='sentiment', color='sentiment', 
                    title=f'Distribusi Sentiment BSI Mobile {PERIOD}', 
                    color_discrete_map= {"negative": "#ff0000", "positive":"#3EA5A1"})
        
        fig.update_traces(textfont=dict(color="black"))

        st.plotly_chart(fig, theme="streamlit")
        st.write()
        st.markdown(f'<span style="font-size: 18px;">:green[Insight Sentiment Pie Chart]</span>', unsafe_allow_html=True)
        st.write("Ulasan BSI Mobile di dominasi oleh ulasan dengan sentiment positive sebesar 66%. Silahkan lakukan filter tanggal untuk melihat distribusi pada range waktu yang ditentukan.")
    with kol2:
        fig = px.pie(dist_score, values='jumlah', names='rating', color='rating', 
                    title=f'Distribusi Rating BSI Mobile {PERIOD}', 
                    color_discrete_map= {"1": "#ff0000", "2": "#ff6666", "3":"#ff9999", "4":"#9ed2d0", "5":"#3EA5A1"})

        fig.update_traces(textfont=dict(color="black"))
        st.plotly_chart(fig, theme="streamlit")
        st.markdown(f'<span style="font-size: 18px;">:green[Insight Rating Pie Chart]</span>', unsafe_allow_html=True)
        st.write("Rating BSI Mobile didominasi dengan rating 5 sebesar 63,3%, sedangkan rating 4 sebesar 3,24% jika dijumlahkan menjadi 66,54%. Jika diasumsikan rating 5 dan 4 cenderung positif, ini sangat mirip dengan hasil analisis sentiment dengan distribusi yang hampir mirip.")

    st.subheader("Sentiment BSI Mobile in Time Series")
    sentiment_bulan = df.groupby(['tahun_bulan_01', 'sentiment']).size().unstack(fill_value=0).reset_index()
    if 'negative' not in sentiment_bulan.columns:
            sentiment_bulan['negative'] = 0
    if 'positive' not in sentiment_bulan.columns:
        sentiment_bulan['positive'] = 0
    
    sentiment_bulan.rename({'tahun_bulan_01': 'Periode'}, axis=1, inplace=True)
    sentiment_bulan = pd.melt(sentiment_bulan, id_vars=['Periode'], var_name='Sentiment', value_name='Nilai')

    fig = px.bar(sentiment_bulan, x="Periode", y="Nilai",
             color="Sentiment",
             barmode = 'group', color_discrete_map={'positive':'#3EA5A1', 'negative':'#ff0000'})
    st.plotly_chart(fig, theme="streamlit")
    web_scrap = st.selectbox(label="Web Scrapping Kulminasi Sentiment", 
                             options=["Desember 2023", "Mei 2023"])
    if web_scrap == "Desember 2023":
        st.markdown(":green[Web Scrap BSI Desember 2023]")
        with open('dataset/bsi_desember_2023.json', 'rb') as f:
            vocab = json.load(f)
        organic_results = vocab.get('organic_results', [])
        or_df = pd.DataFrame(organic_results, columns=["date", "description", "displayed_link", 
                                                                    "domain", "link", "rating", 
                                                                    "summary", 
                                                                    "title"])[["title", "description", "link"]]
        st.dataframe(or_df)
    else:
        st.markdown(":red[Web Scrap BSI Mei 2023]")
        with open('dataset/bsi_mei_2023.json', 'rb') as f:
            vocab = json.load(f)
        organic_results = vocab.get('organic_results', [])
        or_df = pd.DataFrame(organic_results, columns=["date", "description", "displayed_link", 
                                                                    "domain", "link", "rating", 
                                                                    "summary", 
                                                                    "title"])[["title", "description", "link"]]
        st.dataframe(or_df)
    
    # Stack Chart untuk antar tahun 
    warna = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf',
    # blue-teal
]
    if "positive" in positiveNegative:
        st.subheader(f"Sentiment BSI Mobile antar tahun\n{PERIOD}")
        persentase_sentiment_bulan = df.groupby(['tahun', 'bulan', 'sentiment']).size().unstack(fill_value=0).reset_index()
        if 'negative' not in persentase_sentiment_bulan.columns:
            persentase_sentiment_bulan['negative'] = 0
        if 'positive' not in persentase_sentiment_bulan.columns:
            persentase_sentiment_bulan['positive'] = 0
        persentase_sentiment_bulan["persentase_positive"] = persentase_sentiment_bulan["positive"]/(persentase_sentiment_bulan["positive"]+persentase_sentiment_bulan["negative"])
        persentase_sentiment_bulan = persentase_sentiment_bulan.drop(["positive", "negative"], axis=1)
        fig = px.line(persentase_sentiment_bulan, x='bulan', y='persentase_positive', 
                        color='tahun', color_discrete_sequence=warna)
        st.plotly_chart(fig)
    
    st.subheader(f"Sentiment Positive VS Negative Antar Jam BSI Mobile\n{PERIOD}")
    col1, col2 = st.columns((5, 1))
    with col1:
        hourly = df.groupby(["jam", "sentiment"]).size().unstack(fill_value=0).reset_index()
        if 'negative' not in hourly.columns:
            hourly['negative'] = 0
        if 'positive' not in hourly.columns:
            hourly['positive'] = 0
        fig = px.line(hourly, x ='jam', y=["negative", "positive"], 
                      color_discrete_map={'positive':'#3EA5A1', 'negative':'#ff0000'})
        fig.update_xaxes(tickangle=90)
        st.plotly_chart(fig, theme="streamlit")
        # st.write("Waktu Indonesia Barat UTC+7")
    with col2:
        st.markdown(f'<span style="font-size: 18px;">:green[Insight Hourly Sentiment Chart]</span>', unsafe_allow_html=True)
        st.write("Reviewer cenderung memberikan ulasan pada pagi hingga siang hari dengan titik tertinggi pada jam 4 pagi. Dimana saat itu, didominasi dengan ulasan ber-sentiment positif. Sedangkan pada sentiment negatif, reviewer memberikan ulasannya kebanyakan pada pukul 2-12 siang. Memungkinkan BSI Mobile memiliki banyak masalah saat digunakan pada rentang waktu tersebut")

    st.subheader(f"Sentiment Positive VS Negative Antar Bulan BSI Mobile\n{PERIOD}")
    col1, col2 = st.columns((5, 1))
    with col1:
        monthly = df.groupby(["bulan", "sentiment"]).size().unstack(fill_value=0).reset_index()
        if 'negative' not in monthly.columns:
            monthly['negative'] = 0
        if 'positive' not in monthly.columns:
            monthly['positive'] = 0
        fig = px.line(monthly, x ='bulan', y=["negative", "positive"],color_discrete_sequence= ["#ff0000", "#3EA5A1"])
        st.plotly_chart(fig, theme="streamlit")
    with col2:
        st.markdown(f'<span style="font-size: 18px;">:green[Insight Monthly Sentiment Chart]</span>', unsafe_allow_html=True)
        st.write(":green[Pada Sentiment Positive], ulasan terlihat memiliki 2 kali periode, yaitu pada akhir tahun sebelumnya-awal tahun dan tengah tahun dengan kata lain pada akhir semester.")
        st.write(":red[Pada Sentiment negative], ulasan cederung simetris dengan titik puncak pada bulan Mei [5], silahkan filter bulan 5 untuk mendapat insight.")
    
    st.subheader(f"Sentiment Positive VS Negative Antar Tanggal BSI Mobile\n{PERIOD}")
    col1, col2 = st.columns((5, 1))
    with col1:
        date_day = df.groupby(["date_day", "sentiment"]).size().unstack(fill_value=0).reset_index()
        if 'negative' not in date_day.columns:
            date_day['negative'] = 0
        if 'positive' not in date_day.columns:
            date_day['positive'] = 0
        fig = px.line(date_day, x ='date_day', y=["negative", "positive"], color_discrete_map= {"negative": "#ff0000", "positive":"#3EA5A1"})
        st.plotly_chart(fig, theme="streamlit")
    with col2:
        st.markdown(f'<span style="font-size: 18px;">:green[Insight Date Sentiment Chart]</span>', unsafe_allow_html=True)
        st.write("Ulasan ber-:red[sentiment negatif] cenderung banyak terjadi pada tanggal 1 dan rentang tanggal 7-11, sedangkan :green[sentiment positive] cenderung fluktuatif dengan titik puncak pada tanggal 9.")

    st.subheader(f"Sentiment Positive VS Negative Antar Hari BSI Mobile\n{PERIOD}")
    col1, col2 = st.columns((4, 1))
    with col1:
        day_name = df.groupby(["dayNames", "sentiment"]).size().unstack(fill_value=0).reset_index()
        custome_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_name["hariKe"] = np.where(day_name["dayNames"]=="Monday", 0, 
                             np.where(day_name["dayNames"]=="Tuesday", 1, 
                             np.where(day_name["dayNames"]=="Wednesday", 2,
                             np.where(day_name["dayNames"]=="Thursday", 3,
                             np.where(day_name["dayNames"]=="Friday", 4,
                             np.where(day_name["dayNames"]=="Saturday", 5, 6 )
                             ))) ))
        day_name["Custome_Day_Names"] = day_name["hariKe"].astype(str) + "_" + day_name["dayNames"]
        rubah = day_name.sort_values(by="hariKe")
        if 'negative' not in rubah.columns:
            rubah['negative'] = 0
        if 'positive' not in rubah.columns:
            rubah['positive'] = 0
        fig = px.line(rubah, x ='Custome_Day_Names', y=["negative", "positive"],color_discrete_sequence= ["#ff0000", "#3EA5A1"])
        st.plotly_chart(fig, theme="streamlit")
    with col2:
        st.markdown(f'<span style="font-size: 18px;">:green[Insight Daily Sentiment Chart]</span>', unsafe_allow_html=True)
        st.write("Ulasan banyak diberikan pada Hari Kerja / Weekdays (Senin-Jumat) dimana baik :green[sentiment positive] maupun :red[negative] cenderung memiliki arah yang sama.")
    ## Wordcloud
    st.header("WordCloud BSI Mobile")

    STOP_WORDS = []
    with open("stop_wordcloud.txt", "r") as infile:
        STOP_WORDS = infile.read().splitlines()
    
    maks_kata = st.select_slider("Jumlah Maksimum Kata", options=range(1, 1000), value=200)
    cola, colb, colc = st.columns(3)
    with cola:
        staticInteractive = st.selectbox("Static or Interactive Chart", options=["Static", "Interactive"])
    with colb:
        lowerOrUpper = st.selectbox("Kapitalisasi Kata", options=["Lower", "Upper"])
    with colc:
        background = st.color_picker("Background Warna", value="#C5F5F0")
    

    def Prepcloudofword(lowerOrUpper):
        word_cloud = df[df["sentences_wordCloud"].notna()]
        word_string = " ".join(word_cloud['sentences_wordCloud'])
        if lowerOrUpper == "Lower":
            word_string = word_string.lower()
        else:
            word_string = word_string.upper()
        return word_string

    def staticWC(word_string, bg):
        wc = WordCloud(background_color=bg, stopwords = STOP_WORDS, max_words=maks_kata, 
                       max_font_size= 300,  width=1600, height=800)
        wc.generate(word_string)
        return wc
    
    if staticInteractive == "Static":
        word_string = Prepcloudofword(lowerOrUpper)
        wc = staticWC(word_string, background)
        st.image(wc.to_array(), use_column_width=True)
    elif staticInteractive == "Interactive":
        st.markdown("""Mohon maaf, karena library ini sedang bermasalah, terpaksa fitur ini diberhentikan karena tidak compatible dengan library yang lain.
                    Sebelumnya pada Lokal, fungsi ini cukup dapat digunakan walaupun masih ada bug. """)

    
    ## Contoh Sentiment Positive Negative
    st.header(f"Contoh Ulasan Bersentiment Positive VS Negative")
    st.write(f"dalam {PERIOD}")

    positive = positive.sort_values(by="prob_keyakinan", ascending=False)["Sentences"].head(20).reset_index()
    negative = negative.sort_values(by="prob_keyakinan", ascending=True)["Sentences"].head(20).reset_index()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(":green[Positive]")
        def highlight_positive(row):
            return ['background-color: #c5e4e2']*len(row)
        if not positive.empty:
            positive_styled = positive.style.apply(highlight_positive, axis=1)
            st.dataframe(positive_styled)
    with col2:
        st.subheader(":red[Negative]")
        def highlight_negative(row):
            return ['background-color: #ff7f7f']*len(row)
        if not negative.empty:
            negative_styled = negative.style.apply(highlight_negative, axis=1)
            st.dataframe(negative_styled)
    st.subheader("Ulasan Terbaru")
    st.write(f"pada {PERIOD}")
    ulasanTerbaru = df.sort_values(by = "datetime_baru", ascending=False)[["datetime_baru", "Sentences", "sentiment"]].head(10).reset_index(drop=True)
    ulasanTerbaru.columns = ["timestamp", "ulasan", "sentiment"]
    def highlight_sentiment(s):
        return ['background-color: #c5e4e2']*len(s) if s.sentiment =="positive" else ['background-color: #ff7f7f']*len(s)
    
    st.dataframe(ulasanTerbaru.style.apply(highlight_sentiment, axis=1))

with tab2:
    st.header("Perbandingan Performa Ulasan BSI dengan Bank Lainnya pada Play Store per Agustus 2024")
    bb = pd.read_csv("dataset/bank_terbesar_terpilih.csv")
    bb = bb[["developer", "nama_aplikasi", 'jumlah_terinstall', 'banyak_rating', 'banyak_pengulas', 'rerata_rating']]
    bs = pd.read_csv("dataset/bank_syariah_terpilih.csv")
    bs = bs[["developer", "nama_aplikasi", 'jumlah_terinstall', 'banyak_rating', 'banyak_pengulas', 'rerata_rating']]

    ### bank terbesar
    st.subheader("Ulasan BSI dibandingkan 5 Bank Besar Lainnya")
    # rerata tiap indikator 
    rerata_install_b = np.mean(bb["jumlah_terinstall"])
    rerata_banyak_rating_b = np.mean(bb["banyak_rating"])
    rerata_banyak_pengulas_b = np.mean(bb["banyak_pengulas"])
    rerata_rerata_rating_b = np.mean(bb["rerata_rating"])

    # fungsi background rerata install
    def install_b(val):
        color = '#D3F6EC' if val>rerata_install_b else 'white'
        return f'background-color: {color}'
    # fungsi background rerata banyak rating
    def perating_b(val):
        color = '#A8E8D1' if val>rerata_banyak_rating_b else 'white'
        return f'background-color: {color}'
    # fungsi background rerata banyak rating
    def pengulas_b(val):
        color = '#B4F3E2' if val>rerata_banyak_pengulas_b else 'white'
        return f'background-color: {color}'
    # fungsi background rerata rating
    def rating_b(val):
        color = '#C5F5F0' if val>=rerata_rerata_rating_b else 'white'
        return f'background-color: {color}'
    
    besar = bb.style.applymap(install_b, subset=['jumlah_terinstall'])
    besar = besar.applymap(perating_b, subset=['banyak_rating'])
    besar = besar.applymap(pengulas_b, subset=['banyak_pengulas'])
    besar = besar.applymap(rating_b, subset=['rerata_rating'])
    tabel, tr = st.columns((4, 1))
    with tabel:
        st.dataframe(besar)
        st.write(":red[Note]: Cell dengan Highlight adalah Cell dengan Nilai Diatas Rata-rata Nilai Kolom.")
    with tr:
        st.write("Top Bank berdasar Indikator")
        bri = Image.open("image/bri.png")
        st.image(bri)
    ### Bank Syariah 
    st.subheader("Ulasan BSI dibandingkan 5 Bank Syariah Lainnya")
    # rerata tiap indikator 
    rerata_install_s = np.mean(bs["jumlah_terinstall"])
    rerata_banyak_rating_s = np.mean(bs["banyak_rating"])
    rerata_banyak_pengulas_s = np.mean(bs["banyak_pengulas"])
    rerata_rerata_rating_s = np.mean(bs["rerata_rating"])

    # fungsi background rerata install
    def install_s(val):
        color = '#D3F6EC' if val>rerata_install_s else 'white'
        return f'background-color: {color}'
    # fungsi background rerata banyak rating
    def perating_s(val):
        color = '#A8E8D1' if val>rerata_banyak_rating_s else 'white'
        return f'background-color: {color}'
    # fungsi background rerata banyak rating
    def pengulas_s(val):
        color = '#B4F3E2' if val>rerata_banyak_pengulas_s else 'white'
        return f'background-color: {color}'
    # fungsi background rerata rating
    def rating_s(val):
        color = '#C5F5F0' if val>=rerata_rerata_rating_s else 'white'
        return f'background-color: {color}'
    
    syariah = bs.style.applymap(install_s, subset=['jumlah_terinstall'])
    syariah = syariah.applymap(perating_s, subset=['banyak_rating'])
    syariah = syariah.applymap(pengulas_s, subset=['banyak_pengulas'])
    syariah = syariah.applymap(rating_s, subset=['rerata_rating'])
    tabel, tr = st.columns((4, 1))
    with tabel:
        st.dataframe(syariah)
        st.write(":red[Note]: Cell dengan Highlight adalah Cell dengan Nilai Diatas Rata-rata Nilai Kolom.")
    with tr:
        st.write("Top Bank berdasar Indikator")
        st.image(logo)

with tab3:
    with st.container():
        text_column, mid, image_column = st.columns((1,0.2,0.5))
        with text_column:
            # kl1, kl2, kl3 = st.columns(3)
            # # with kl2:
            # #     st.title("Thank You")
            # st.write("This Dashboard Build by Support:")
            stream = Image.open("image/streamlit.png")

            st.subheader("Visualization", divider='green')
            plotly = Image.open("image/plotly.png")
            altair = Image.open("image/altair.png")
            altair = Image.open("image/altair.png")
            k1, k2, k3 = st.columns(3)
            with k1:
                st.image(plotly)
            with k2:
                st.image(altair)
            with k3:
                st.image(stream)

            st.subheader("Modelling", divider='green')
            keras = Image.open("image/keras.png")
            tensorf = Image.open("image/tf.png")
            nltk = Image.open("image/nltk.png")
            sklearn = Image.open("image/sklearn.png")
            
            k1, k2 = st.columns(2)
            with k1:
                st.image(keras)
            with k2:
                st.image(nltk)
            k1, k2 = st.columns(2)
            with k1:
                st.image(tensorf)
            with k2:
                st.image(sklearn)

            st.subheader("Preprocessing", divider='green')
            pan_das = Image.open("image/pd.png")
            num_py = Image.open("image/np.png")
            fast_text = Image.open("image/fasttext.png")
            k1, k2, k3 = st.columns(3)
            with k1:
                st.image(pan_das)
            with k2:
                st.image(num_py)
            with k3:
                st.image(fast_text)

            st.subheader("Special Thanks", divider='green')
            st.markdown("""
                        - BSI Data Analytics Departement (lead by Bapak Denny Setiawan), Specially for Digital Team (lead by Mbak Ida)\n
                        - [Google Play Scraper](https://github.com/JoMingyu/google-play-scraper)
                        - [Zenrows](https://www.zenrows.com/)
                        """)
          
            st.subheader("Info Update", divider='red')
            st.write(":red[Next Development]: Topic Generation :green[Done], Semi Real Time :green[Done]")

            st.write("Let's connect! You may either reach out to me at ludy.hasby@gmail.com or learn me more at [this link](https://ludyhasby.streamlit.app/)")

        with image_column:
            st.subheader("Ludy Hasby Aulia")
            st.markdown(":green[Aspiring Data Scientist/Software Engineer also studies Economics]")
            # st.write("Computer Science - Economics")
            linkedin_url = "https://www.linkedin.com/in/ludy-hasby/"
            github_url = "https://github.com/ludyhasby"
            email_url = "mailto:ludy.hasby@gmail.com"
            st.markdown(
                social_icons(32, 32, LinkedIn=linkedin_url, GitHub=github_url, Email=email_url),
                unsafe_allow_html=True)

st.markdown("*Copyright © 2024 Ludy Hasby Aulia*")