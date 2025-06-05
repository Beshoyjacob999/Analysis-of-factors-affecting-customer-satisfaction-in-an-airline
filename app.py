import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from matplotlib.colors import LinearSegmentedColormap
page_bg_color = """
<style>
[data-testid="stAppViewContainer"] { background-color: #626F47; }
[data-testid="stSidebar"] { background-color: #A4B465; }
h1, h2, h3, h4, h5, h6 { color: #FFCF50; }
select, button { background-color: #626F47 !important; color: #FFCF50 !important; border-radius: 10px !important; }
body, p, div { color: #FFCF50; }
header, [data-testid="stHeader"] { background-color: #626F47 !important; }
div[data-testid="stSelectbox"] div[role="combobox"],
div[data-testid="stSelectbox"] div[data-testid="stDropdownContainer"],
div[data-testid="stSelectbox"] div[data-testid="stDropdownContainer"] div {
    background-color: black !important;
    color: white !important;
    border-radius: 5px !important;
}
</style>
"""
custom_cmap = LinearSegmentedColormap.from_list("custom_green", ["#FEFAE0", "#4A703B"])
st.markdown(page_bg_color, unsafe_allow_html=True)

def style_plot(title=None, xlabel=None, ylabel=None):
    ax = plt.gca()
    fig = plt.gcf()
    ax.set_facecolor('#626F47')
    fig.patch.set_facecolor('#626F47')
    ax.tick_params(axis='x', colors='#FFCF50')
    ax.tick_params(axis='y', colors='#FFCF50')
    ax.xaxis.label.set_color('#FFCF50')
    ax.yaxis.label.set_color('#FFCF50')
    if title: ax.set_title(title, color='#FFCF50')
    if xlabel: ax.set_xlabel(xlabel, color='#FFCF50')
    if ylabel: ax.set_ylabel(ylabel, color='#FFCF50')

df = pd.read_csv('df_copy.csv', encoding='ISO-8859-1')
df.dropna(inplace=True)
df['satisfaction_num'] = df['satisfaction'].replace({'neutral or dissatisfied': 0, 'satisfied': 1})
df['class_num'] = df['class'].replace({'Business': 1, 'Eco Plus': 2, 'Eco': 3})
df['type_of_travel_num'] = df['type_of_travel'].replace({'Business travel': 1, 'Personal Travel': 0})

features = ['class_num', 'type_of_travel_num', 'food_and_drink', 'online_boarding',
            'seat_comfort', 'inflight_entertainment', 'on-board_service', 'leg_room_service',
            'baggage_handling', 'checkin_service', 'inflight_service', 'cleanliness',
            'departure_delay_in_minutes', 'arrival_delay_in_minutes']

x = df[features]
y = df['satisfaction_num']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=40)
model = DecisionTreeClassifier(random_state=40)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

feature_importance_dict = dict(zip(x_train.columns, model.feature_importances_))
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)


def main_page():
    st.title("Services Analysis")
    num_sorted_feature_importance = st.sidebar.slider('Select number of features to display', 1, 14, 7)

    if st.sidebar.button('Check'):
        features_selected = [item[0] for item in sorted_feature_importance[:num_sorted_feature_importance]]
        importances = [item[1] for item in sorted_feature_importance[:num_sorted_feature_importance]]
        custom_cmap = LinearSegmentedColormap.from_list("custom_green", ["#FEFAE0", "#4A703B"])
        colors = [custom_cmap(i / (num_sorted_feature_importance - 1)) for i in range(num_sorted_feature_importance)]
        colors.reverse()

        plt.figure(figsize=(10, 5))
        plt.bar(features_selected, importances, color=colors)
        style_plot("Top Feature Importances", "Features", "Importance")
        plt.xticks(rotation=90)
        st.pyplot(plt)

        mean_df = df.groupby('satisfaction')[['cleanliness','inflight_wifi_service','ease_of_online_booking','food_and_drink',
            'online_boarding', 'seat_comfort','inflight_entertainment','on-board_service','leg_room_service',
            'baggage_handling','checkin_service','inflight_service']].mean()

        ordered_cols = mean_df.mean().sort_values(ascending=False).index.tolist()
        for level in mean_df.index:
            values = mean_df.loc[level][ordered_cols]
            plt.figure(figsize=(10, 5))
            custom_cmap = LinearSegmentedColormap.from_list("custom_green", ["#FEFAE0", "#4A703B"])
            colors = [custom_cmap(i / (num_sorted_feature_importance - 1)) for i in range(num_sorted_feature_importance)]
            colors.reverse()
            plt.bar(values.index, values.values, color=colors)
            style_plot(f"Satisfaction Level: {level}", "Service", "Average Score")
            plt.xticks(rotation=90)
            plt.tight_layout()
            st.pyplot(plt)

def second_page():
    st.title("Satisfaction Analysis")
    col1, col2 = st.columns(2)


    plt.figure(figsize=(8, 6))
    df.groupby('type_of_travel')['satisfaction_num'].mean().plot(kind='bar', color=sns.color_palette("YlGn", 2))
    style_plot("Customer Satisfaction by Type of Travel", "Type of Travel", "Satisfaction")
    st.pyplot(plt)

    with col1:

        satisfaction_counts = df['satisfaction'].value_counts()
        custom_cmap = LinearSegmentedColormap.from_list("custom_green", ["#A4B465", "#FEFAE0"])
        colors = [custom_cmap(i / (len(satisfaction_counts) - 1)) for i in range(len(satisfaction_counts))]
        fig = plt.figure(figsize=(6, 6))
        fig.patch.set_facecolor('#626F47')
        plt.pie(satisfaction_counts, labels=satisfaction_counts.index, autopct='%1.1f%%', colors=colors)
        plt.title("Satisfaction Distribution", color='#FFCF50')
        st.pyplot(plt)

        custom_colors = ["#A4B465", "#FEFAE0"]
        plt.figure(figsize=(10, 6))
        plt.gca().set_facecolor('#626F47')
        plt.gcf().set_facecolor('#626F47')
        sns.kdeplot(data=df, x='age', hue='satisfaction', fill=True,common_norm=False, palette=custom_colors)
        plt.title("Age Distribution by Satisfaction", color='#FFCF50')
        plt.xlabel('Age', color='#FEFAE0')
        plt.ylabel('Density', color='#FEFAE0')
        plt.tick_params(colors='#FEFAE0')
        st.pyplot(plt)

    with col2:

        plt.figure(figsize=(8, 6))

        df.groupby('customer_type')['satisfaction_num'].mean().plot(kind='bar', color=sns.color_palette("YlGn", 2))
        style_plot("Satisfaction by Customer Type", "Customer Type", "Satisfaction")
        st.pyplot(plt)


        gender_counts = df['gender'].value_counts()
        custom_cmap = LinearSegmentedColormap.from_list("custom_green", ["#A4B465", "#FEFAE0"])
        colors = [custom_cmap(i / (len(gender_counts) - 1)) for i in range(len(gender_counts))]
        fig = plt.figure(figsize=(6, 6))
        fig.patch.set_facecolor('#626F47')
        plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',colors=colors)
        plt.title("Gender Distribution")
        st.pyplot(plt)

page = st.sidebar.selectbox("Select the type of analysis", ["Services Analysis", "Satisfaction  analysis"])
if page == "Services Analysis":
    main_page()
else:
    second_page()
