import os
import json
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from collections import Counter
from pyvis.network import Network

# Step 1: Load the data
df = pd.read_csv('data/ptt_ai_221130to240531.csv')
df.head()
df.info()

# Load the font
font_path = 'font/TraditionalChinese.ttf'
font_manager.fontManager.addfont(font_path)
for font in font_manager.fontManager.ttflist:
    if font.fname == font_path:
        print(f"Found font: {font.name}")
        plt.rcParams['font.family'] = font.name
        break

# Step 2: Data Preprocessing
clear_df = df.copy()
drop_cols = ['system_id', 'artTitle', 'artCatagory', 'dataSource', 'insertedDate']
clear_df.drop(drop_cols, axis=1, inplace=True)
clear_df.dropna(subset=['artContent'], axis=0, how='any', inplace=True)
clear_df['sentence'] = clear_df['artContent'].str.replace(r'\n\n', '。', regex=True)
clear_df['sentence'] = clear_df['sentence'].str.replace(r'\n', '，', regex=True)
clear_df['sentence'] = clear_df['sentence'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
clear_df.head(10)

# Step 3: Deal with Comments
tqdm.pandas()


def get_comment_info(com):
    commenters, comment_status = [], []
    com = eval(com)
    for i in com:
        commenters.append(i['cmtPoster'])
        comment_status.append(i['cmtStatus'])
    return pd.Series([commenters, comment_status])


clear_df[['commenters', 'comment_status']] = clear_df['artComment'].apply(get_comment_info)
clear_df.head()

# Step 4: Explode the DataFrame
clear_df = clear_df.explode(['commenters', 'comment_status'])
social_df = clear_df[['artPoster', 'artUrl', 'commenters', 'comment_status']]
social_df.head()
social_df['comment_status'].value_counts()

# Step 5: Filter the Data
user_count = Counter(social_df['artPoster'].tolist() + social_df['commenters'].tolist())
top_users = {user for user, count in user_count.most_common(30)}
top_filtered_df = social_df[social_df['artPoster'].isin(top_users) & social_df['commenters'].isin(top_users)]
top_filtered_df.head()

# Step 6: Transform the Data
re_df = top_filtered_df[['commenters', 'artUrl', 'comment_status']].rename(
    columns={'commenters': 'src', 'artUrl': 'dis', 'comment_status': 'weight'})
re_df = re_df[~re_df['src'].isna()]
re_df.head()


def convert_status(s):
    if s == '推':
        return 2
    elif s == '→':
        return 1
    else:
        return -1


re_df['weight'] = re_df['weight'].map(convert_status)
re_df = re_df.groupby(['src', 'dis']).sum().reset_index()


# Get edge color based on weight
def get_color(w):
    if w > 0:
        return 'green'
    else:
        return 'red'


re_df['color'] = re_df['weight'].map(get_color)
re_df

# Add post data for network creation
po_df = top_filtered_df[['artPoster', 'artUrl']].rename(columns={'artPoster': 'src', 'artUrl': 'dis'})

# Step 7: Create the Network
netWork = Network(notebook=True, cdn_resources='in_line', directed=True)
person = list(set(po_df.src.unique().tolist() + re_df.src.unique().tolist()))
url = po_df.dis.unique().tolist()

# Add nodes (people)
netWork.add_nodes(
    nodes=person,
    value=[1 for _ in range(len(person))],
    color=['#66CDAA' for _ in range(len(person))],
    title=person
)

# Add nodes (articles)
netWork.add_nodes(
    nodes=url,
    value=[2 for _ in range(len(url))],
    color=['#FFB366' for _ in range(len(url))],
    title=url
)

# Add edges (posters -> articles)
for i in po_df.to_numpy():
    netWork.add_edge(i[0], i[1], width=2, color='grey')

# Add edges (commenters -> articles) with color based on weight
for i in re_df.to_numpy():
    netWork.add_edge(i[0], i[1], width=2, color=i[3])

# Set layout for better visualization
netWork.repulsion()

# Create plot directory if not exists
if not os.path.exists('plot'):
    os.makedirs('plot')

# Save and show the interactive network graph
netWork.show('plot/Network.html')
