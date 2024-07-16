from flask import Flask, jsonify, render_template, request
import json
import pandas as pd
import re
from datetime import timedelta
import matplotlib
matplotlib.use('Agg')  # Utilisation du backend Agg pour Matplotlib
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

from pymongo import MongoClient

# Configuration de la connexion MongoDB Atlas
client = MongoClient("mongodb+srv://anass:anass@cluster0.oqk5dfg.mongodb.net/")
db = client['Ouilog']
collection = db['mouvements2024']


# Fonction pour charger et traiter les données du premier script
def load_and_process_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Utilisateur'] = df['Utilisateur'].fillna('Inconnu')
    ramassage = df[(df['Emplacement'].str.startswith('CA')) & (df['Commentaire'].str.startswith('Déplacement'))]
    eclatage = df[(df['Emplacement'].str.startswith('BA')) & (df['Commentaire'].str.startswith('Déplacement'))]
    colisage = df[df['Emplacement'].str.contains('CO') & (df['Commentaire'].str.startswith('Déplacement'))]

    def calculate_durations_with_condition(df, time_delta):
        results = []
        grouped = df.groupby('Utilisateur')
        for user, group in grouped:
            group = group.sort_values(by='Date').reset_index(drop=True)
            group['Duration'] = group['Date'].diff().fillna(pd.Timedelta(seconds=0))
            group['NewSegment'] = (group['Duration'] > time_delta).cumsum()
            segments = group.groupby('NewSegment').apply(lambda x: x['Date'].iloc[-1] - x['Date'].iloc[0])
            for segment in segments:
                results.append({'Utilisateur': user, 'Duration': segment})
        return pd.DataFrame(results)

    ramassage_durations = calculate_durations_with_condition(ramassage, timedelta(minutes=10))
    eclatage_durations = calculate_durations_with_condition(eclatage, timedelta(minutes=10))
    colisage_durations = calculate_durations_with_condition(colisage, timedelta(minutes=15))

    def filter_series_orders(df, time_threshold=pd.Timedelta(minutes=0)):
        df = df.sort_values(by='Date').reset_index(drop=True)
        df['NextOrderDiff'] = df['Date'].diff().shift(-1).fillna(pd.Timedelta(seconds=0))
        df['SeriesOrder'] = (df['NextOrderDiff'] <= time_threshold)
        filtered_df = df[~df['SeriesOrder']]
        return filtered_df

    colisage_filtered = filter_series_orders(colisage)
    colisage_durations = calculate_durations_with_condition(colisage_filtered, timedelta(minutes=15))

    def summarize_durations(durations, original_df):
        grouped = durations.groupby('Utilisateur')
        stats = []
        for user, group in grouped:
            total_duration = group['Duration'].sum()
            min_duration = group[group['Duration'] != timedelta(0)]['Duration'].min()
            max_duration = group['Duration'].max()
            user_orders = original_df[original_df['Utilisateur'] == user]['Num cde client'].nunique()
            user_orders = max(user_orders, 1)
            mean_time_per_order = total_duration / user_orders
            stats.append({
                'Utilisateur': user,
                'TotalDuration': total_duration,
                'MinDuration': min_duration if pd.notna(min_duration) else timedelta(0),
                'MaxDuration': max_duration,
                'Orders': user_orders,
                'MeanTimePerOrder': mean_time_per_order
            })
        return pd.DataFrame(stats)

    ramassage_stats = summarize_durations(ramassage_durations, ramassage)
    eclatage_stats = summarize_durations(eclatage_durations, eclatage)
    colisage_stats = summarize_durations(colisage_durations, colisage_filtered)

    def calculate_global_stats(stats_df):
        total_duration = stats_df['TotalDuration'].sum()
        min_duration = stats_df['MinDuration'].min()
        max_duration = stats_df['MaxDuration'].max()
        total_orders = stats_df['Orders'].sum()
        mean_time_per_order = total_duration / total_orders if total_orders > 0 else timedelta(0)
        return {
            'TotalDuration': total_duration,
            'MinDuration': min_duration,
            'MaxDuration': max_duration,
            'TotalOrders': total_orders,
            'MeanTimePerOrder': mean_time_per_order
        }

    ramassage_global_stats = calculate_global_stats(ramassage_stats)
    eclatage_global_stats = calculate_global_stats(eclatage_stats)
    colisage_global_stats = calculate_global_stats(colisage_stats)

    result = {
        'ramassage_stats': ramassage_stats.to_dict(orient='records'),
        'eclatage_stats': eclatage_stats.to_dict(orient='records'),
        'colisage_stats': colisage_stats.to_dict(orient='records'),
        'ramassage_global_stats': ramassage_global_stats,
        'eclatage_global_stats': eclatage_global_stats,
        'colisage_global_stats': colisage_global_stats
    }

    return result

def load_and_process_data_2(df):
    def filter_total_references(df):
        def extract_unique_identifier(commentaire):
            match = re.search(r'(E|R|ZM)-\d-\d+-H\d-\d', commentaire)
            return match.group(0) if match else None

        total_df = df[(df['Emplacement'].str.startswith('CA')) & (df['Commentaire'].str.startswith('Déplacement'))]
        total_df['UniqueIdentifier'] = total_df['Commentaire'].apply(extract_unique_identifier)
        total_df = total_df[total_df['UniqueIdentifier'].ne(total_df['UniqueIdentifier'].shift())]
        total_df = total_df.drop(columns=['UniqueIdentifier'])
        return total_df

    def filter_references_ramassees_en_r(df):
        def extract_unique_identifier(commentaire):
            match = re.search(r'R-\d-\d+-H\d-\d', commentaire)
            return match.group(0) if match else None

        df = df[(df['Emplacement'].str.startswith('CA')) & (df['Commentaire'].str.startswith('Déplacement')) & (df['Commentaire'].str.contains(' de R-', na=False))]
        df['UniqueIdentifier'] = df['Commentaire'].apply(extract_unique_identifier)
        df = df[df['UniqueIdentifier'].ne(df['UniqueIdentifier'].shift())]
        df = df.drop(columns=(['UniqueIdentifier']))
        return df

    total_references = filter_total_references(df)
    references_ramassees_en_r = filter_references_ramassees_en_r(df)

    def calculate_percentage(total_df, filtered_df):
        total_references = total_df.shape[0]
        filtered_references = filtered_df.shape[0]
        return (filtered_references / total_references) * 100 if total_references > 0 else 0

    percentage_ramassees_en_r = calculate_percentage(total_references, references_ramassees_en_r)
    # retourner aussi les titre des article ramassées en R
    # print(references_ramassees_en_r["Titre"])
    return {'percentage_ramassees_en_r': percentage_ramassees_en_r}

def load_and_process_data_3(df):
    def filter_references_ramassees_en_e(df):
        return df[(df['Emplacement'].str.startswith('CA')) & (df['Commentaire'].str.startswith('Déplacement')) & (df['Commentaire'].str.contains(' de E-', na=False))]

    references_ramassees_en_e = filter_references_ramassees_en_e(df)

    def extract_location_info(commentaire):
        match = re.search(r'E-(\d)-(\d+)-H(\d)-(\d)', commentaire)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
        return None

    def add_group_identifier(df):
        df['Location'] = df['Commentaire'].apply(extract_location_info)
        df = df.sort_values(by='Date').reset_index(drop=True)
        df['Group'] = (df['Date'].diff() > timedelta(minutes=10)).cumsum()
        return df

    references_ramassees_en_e = add_group_identifier(references_ramassees_en_e)

    def calculate_distance_between_points(start, end):
        if start is None or end is None:
            return 0.0

        couloir_start, section_start, _, _ = start
        couloir_end, section_end, _, _ = end

        distance_total = 0.0

        if couloir_start != couloir_end:
            distance_to_couloir_end_1 = section_start * 1.2
            distance_to_couloir_end_2 = (12 - section_start) * 1.2

            if distance_to_couloir_end_1 < distance_to_couloir_end_2:
                distance_within_couloir = distance_to_couloir_end_1
                distance_from_couloir_start = section_end * 1.2
            else:
                distance_within_couloir = distance_to_couloir_end_2
                distance_from_couloir_start = (12 - section_end) * 1.2

            distance_total += distance_within_couloir
            distance_total += 2
            distance_total += distance_from_couloir_start
        else:
            if section_start != section_end:
                distance_sections = abs(section_end - section_start) * 1.2
                distance_total += distance_sections

        return distance_total

    def calculate_distances(grouped_df):
        distances = []
        for name, group in grouped_df:
            group = group.sort_values(by='Date').reset_index(drop=True)
            total_distance = 0.0
            prev_location = None
            for _, row in group.iterrows():
                current_location = row['Location']
                if prev_location is not None:
                    distance = calculate_distance_between_points(prev_location, current_location)
                    total_distance += distance
                prev_location = current_location
            distances.append({'Group': name, 'Distance(m)': total_distance, 'Details': group[['Date', 'Emplacement', 'Commentaire', 'Location']]})
        return distances

    distances = calculate_distances(references_ramassees_en_e.groupby('Group'))
    return {'distances': distances}

def calculate_percentage_commande_passees_avant_13h(df):
    df['Date'] = pd.to_datetime(df['Date'])

    commandes_creees_avant_13h = df[((df['Commentaire'] == "Création de commande (EVO)") | (df['Commentaire'] == "Création de commande (OMS)")) & (df['Date'].dt.time <= pd.Timestamp("13:00").time())].drop_duplicates(subset=['Ek'])

    commandes_expediees = df[(df['Signe mouvement'] == "-") & (df['Commentaire'] == "Décrémentation du stock suite à expédition commande")].drop_duplicates(subset=['Ek'])

    commandes_merged = pd.merge(commandes_creees_avant_13h, commandes_expediees, on="Ek", suffixes=('_creee', '_expediee'))

    commandes_du_jour = commandes_merged[commandes_merged['Date_creee'].dt.date == commandes_merged['Date_expediee'].dt.date]
    commandes_veille = commandes_merged[(commandes_merged['Date_creee'].dt.date == commandes_merged['Date_expediee'].dt.date - pd.Timedelta(days=1)) & (commandes_merged['Date_creee'].dt.time > pd.Timestamp("13:00").time())]

    total_commandes_uniques = df[((df['Commentaire'] == "Création de commande (EVO)") | (df['Commentaire'] == "Création de commande (OMS)"))].drop_duplicates(subset=['Ek'])
    total_commandes = len(total_commandes_uniques)

    pourcentage = (len(commandes_du_jour) + len(commandes_veille)) / total_commandes * 100 if total_commandes > 0 else 0
    return {'percentage_commande_passees_avant_13h': pourcentage}

def plot_global_stats(dates, percentages, label, color):
    img = io.BytesIO()
    plt.figure(figsize=(10, 6))
    plt.plot(dates[:len(percentages)], percentages, marker='o', linestyle='-', color=color)
    plt.xlabel('Date')
    plt.ylabel(f'Pourcentage {label}')
    plt.title(f'Pourcentage {label} par jour')
    plt.grid(True)
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_global_stats2(dates, percentages, averages, label, color):
    img = io.BytesIO()
    plt.figure(figsize=(10, 6))
    plt.plot(dates[:len(percentages)], percentages, marker='o', linestyle='-', color=color, label=f'Courbe {label}')
    plt.plot(dates[:len(averages)], averages, marker='x', linestyle='--', color='gray', label='Nombre moyen d\'articles par commande')
    plt.xlabel('Date')
    plt.ylabel(f'Temps moyen du {label} et Nombre moyen d\'articles par commande')
    plt.title(f'Temps moyen du {label} et Nombre moyen d\'articles par commande par jour')
    plt.grid(True)
    plt.legend()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_user_stats(dates, user_data, label):
    img = io.BytesIO()
    plt.figure(figsize=(12, 8))
    for user, data in user_data.items():
        plt.plot(dates, data, marker='o', linestyle='-', label=f'{user} - {label}')
    plt.xlabel('Date')
    plt.ylabel(f'Temps moyen {label} (minutes)')
    plt.title(f'Temps moyen {label} par utilisateur et par jour')
    plt.grid(True)
    plt.legend()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_distance_stats(dates, avg_distance_per_article, avg_distance_per_order):
    img = io.BytesIO()
    plt.figure(figsize=(12, 8))
    plt.plot(dates, avg_distance_per_article, marker='o', linestyle='-', label='Distance moyenne par article')
    plt.plot(dates, avg_distance_per_order, marker='x', linestyle='--', label='Distance moyenne par commande')
    plt.xlabel('Date')
    plt.ylabel('Distance moyenne (mètres)')
    plt.title('Distance moyenne parcourue par article et par commande par jour')
    plt.grid(True)
    plt.legend()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        date_selection_type = request.form['date_selection_type']
        try:
            # with open('data/mouvements2024_df.json', 'r') as f:
            #     data = json.load(f)
            # Récupérer les données depuis MongoDB
            data = list(collection.find({}))

            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df['Date'])
            df['Utilisateur'] = df['Utilisateur'].fillna('Inconnu')

            if date_selection_type == 'single':
                date_selected = request.form['date']
                date_selected = pd.to_datetime(date_selected).date()
                df = df[df['Date'].dt.date == date_selected]

                if df.empty:
                    raise ValueError("Aucune donnée pour la date sélectionnée.")
                
                results = load_and_process_data(df)
                stats_2 = load_and_process_data_2(df)
                stats_3 = load_and_process_data_3(df)
                stats_4 = calculate_percentage_commande_passees_avant_13h(df)

                return render_template('index.html', date=date_selected, results=results, stats_2=stats_2, stats_3=stats_3, stats_4=stats_4)

            elif date_selection_type == 'range':
                start_date = request.form['start_date']
                end_date = request.form['end_date']
                start_date = pd.to_datetime(start_date).date()
                end_date = pd.to_datetime(end_date).date()

                df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]

                if df.empty:
                    raise ValueError("Aucune donnée pour la plage de dates sélectionnée.")
                
                percentages_ramassees_en_r = []
                percentages_ramassage = []
                percentages_eclatement = []
                percentages_colisage = []

                averages_ramassage = []
                averages_eclatement = []
                averages_colisage = []

                user_ramassage_data = {}
                user_colisage_data = {}
                
                avg_distance_per_article = []
                avg_distance_per_order = []

                dates = pd.date_range(start_date, end_date).date
                for date in dates:
                    daily_df = df[df['Date'].dt.date == date]
                    daily_df = daily_df[daily_df['Commentaire'].str.startswith('Déplacement')]
                    if not daily_df.empty:
                        daily_results = load_and_process_data(daily_df)
                        stats = load_and_process_data_2(daily_df)
                        stats_3 = load_and_process_data_3(daily_df)
                        percentages_ramassees_en_r.append(stats['percentage_ramassees_en_r'])
                        percentages_ramassage.append(daily_results['ramassage_global_stats']['MeanTimePerOrder'].total_seconds() / 60)
                        percentages_eclatement.append(daily_results['eclatage_global_stats']['MeanTimePerOrder'].total_seconds() / 60)
                        percentages_colisage.append(daily_results['colisage_global_stats']['MeanTimePerOrder'].total_seconds() / 60)

                        # Calcul du nombre moyen d'articles par commande en éliminant les doublons
                        def calculate_average_articles(df, condition):
                            filtered_df = df[condition].drop_duplicates(subset=['Ek', 'ReferenceEcommercant'])
                            return filtered_df.groupby('Ek')['ReferenceEcommercant'].count().mean()

                        average_articles_ramassage = calculate_average_articles(daily_df, daily_df['Emplacement'].str.startswith('CA'))
                        average_articles_eclatement = calculate_average_articles(daily_df, daily_df['Emplacement'].str.startswith('BA'))
                        average_articles_colisage = calculate_average_articles(daily_df, daily_df['Emplacement'].str.contains('CO'))

                        averages_ramassage.append(average_articles_ramassage if not pd.isna(average_articles_ramassage) else 0)
                        averages_eclatement.append(average_articles_eclatement if not pd.isna(average_articles_eclatement) else 0)
                        averages_colisage.append(average_articles_colisage if not pd.isna(average_articles_colisage) else 0)
                    
                        # Extraction du temps moyen par utilisateur pour ramassage et colisage
                        for stat in daily_results['ramassage_stats']:
                            user = stat['Utilisateur']
                            mean_time = stat['MeanTimePerOrder'].total_seconds() / 60
                            if user not in user_ramassage_data:
                                user_ramassage_data[user] = [0] * len(dates)
                            user_ramassage_data[user][dates.tolist().index(date)] = mean_time

                        for stat in daily_results['colisage_stats']:
                            user = stat['Utilisateur']
                            mean_time = stat['MeanTimePerOrder'].total_seconds() / 60
                            if user not in user_colisage_data:
                                user_colisage_data[user] = [0] * len(dates)
                            user_colisage_data[user][dates.tolist().index(date)] = mean_time

                        # Calcul des distances moyennes
                        total_distance = sum([d['Distance(m)'] for d in stats_3['distances']])
                        total_articles = daily_df['ReferenceEcommercant'].nunique()
                        total_orders = daily_df['Ek'].nunique()

                        avg_distance_per_article.append(total_distance / total_articles if total_articles > 0 else 0)
                        avg_distance_per_order.append(total_distance / total_orders if total_orders > 0 else 0)
                    
                    else:
                        percentages_ramassees_en_r.append(0)
                        percentages_ramassage.append(0)
                        percentages_eclatement.append(0)
                        percentages_colisage.append(0)

                        averages_ramassage.append(0)
                        averages_eclatement.append(0)
                        averages_colisage.append(0)

                        avg_distance_per_article.append(0)
                        avg_distance_per_order.append(0)

                plot_url_ramassees_en_r = plot_global_stats(dates, percentages_ramassees_en_r, "Ramassées en R", 'b')
                plot_url_ramassage = plot_global_stats2(dates, percentages_ramassage, averages_ramassage, "Ramassage", 'r')
                plot_url_eclatement = plot_global_stats2(dates, percentages_eclatement, averages_eclatement, "Eclatement", 'g')
                plot_url_colisage = plot_global_stats2(dates, percentages_colisage, averages_colisage, "Colisage", 'c')

                plot_url_user_ramassage = plot_user_stats(dates, user_ramassage_data, "Ramassage")
                plot_url_user_colisage = plot_user_stats(dates, user_colisage_data, "Colisage")

                plot_url_distance = plot_distance_stats(dates, avg_distance_per_article, avg_distance_per_order)

                return render_template('index.html', plot_url_ramassees_en_r=plot_url_ramassees_en_r, plot_url_ramassage=plot_url_ramassage, plot_url_eclatement=plot_url_eclatement, plot_url_colisage=plot_url_colisage, plot_url_user_ramassage=plot_url_user_ramassage, plot_url_user_colisage=plot_url_user_colisage, plot_url_distance=plot_url_distance, start_date=start_date, end_date=end_date)
        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
