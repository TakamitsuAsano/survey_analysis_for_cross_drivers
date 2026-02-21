import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import io
import graphviz
import json
import re
from streamlit_gsheets import GSheetsConnection
from st_copy_to_clipboard import st_copy_to_clipboard

# --- 日本語フォント設定 ---
def setup_japanese_font():
    font_path = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
    if os.path.exists(font_path):
        try:
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = 'IPAexGothic'
        except Exception as e:
            st.error(f"フォント設定エラー: {e}")
    else:
        try:
            plt.rcParams['font.family'] = 'Hiragino Sans'
        except:
            pass

setup_japanese_font()

# --- 決定木データをDataFrameに変換する関数 ---
def get_decision_tree_data(clf, feature_names, class_names):
    n_nodes = clf.tree_.node_count
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    impurity = clf.tree_.impurity
    n_node_samples = clf.tree_.n_node_samples
    value = clf.tree_.value
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]
    
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    data = []
    for i in range(n_nodes):
        node_type = "Leaf" if is_leaves[i] else "Node"
        if i == 0: node_type = "Root"
        
        if feature[i] != -2:
            feat_name = feature_names[feature[i]]
            th = threshold[i]
            condition = f"{feat_name} <= {th:.3f}"
        else:
            feat_name = None
            th = None
            condition = None
        
        val = value[i][0]
        class_idx = np.argmax(val)
        pred_class = class_names[class_idx] if class_names is not None else str(class_idx)
        
        row = {
            "Node_ID": i,
            "Depth": node_depth[i],
            "Type": node_type,
            "Condition": condition,
            "Feature": feat_name,
            "Threshold": th,
            "Gini": f"{impurity[i]:.4f}",
            "Samples": n_node_samples[i],
            "Value": str(list(map(int, val))),
            "Predicted_Class": pred_class
        }
        data.append(row)
    
    return pd.DataFrame(data)

# --- コールバック関数群 (Tab7用) ---
def cb_update_cross(idx, col):
    st.session_state["sb_cross_index"] = idx
    st.session_state["sb_cross_col"] = col

def cb_update_reg(tgt, feats):
    st.session_state["sb_reg_target"] = tgt
    st.session_state["sb_reg_feature"] = feats
    st.session_state['ai_msg_reg'] = True

def cb_update_tree(tgt, feats):
    st.session_state["sb_tree_target"] = tgt
    st.session_state["sb_tree_feature"] = feats
    st.session_state['ai_msg_tree'] = True

def cb_update_cluster(feats):
    st.session_state["sb_cluster_features"] = feats
    st.session_state['ai_msg_cluster'] = True

# --- 選択肢の並び順を強制する関数 ---
def enforce_likert_order(df, col_name):
    likert_order_master = [
        "非常にそう思う", "強くそう思う", "とてもそう思う", "そう思う", "かなりそう思う",
        "ややそう思う", "どちらかといえばそう思う", "どちらかと言えばそう思う",
        "どちらともいえない", "どちらとも言えない", "普通",
        "どちらかといえばそう思わない", "どちらかと言えばそう思わない", "あまりそう思わない",
        "そう思わない", "全くそう思わない", "全く思わない"
    ]
    unique_vals = df[col_name].dropna().unique()
    sorter = [x for x in likert_order_master if x in unique_vals]
    remaining = [x for x in unique_vals if x not in sorter]
    sorted_remaining = sorted(remaining, key=lambda x: str(x))
    final_order = sorter + sorted_remaining
    
    if len(sorter) >= 2:
        return pd.Categorical(df[col_name], categories=final_order, ordered=True)
    else:
        return df[col_name]

# ---------------------------------------

st.set_page_config(page_title="アンケート分析 & セグメンテーション", layout="wide")
st.title("📊 アンケート分析 & セグメンテーションツール")

# --- サイドバー：データ読み込み ---
st.sidebar.header("データの読み込み")
input_method = st.sidebar.radio("データの種類を選択", ["ファイルアップロード", "Googleスプレッドシート"])

df = None

if input_method == "ファイルアップロード":
    uploaded_file = st.sidebar.file_uploader("ExcelまたはCSVファイルをアップロード", type=['xlsx', 'csv'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.sidebar.success("ファイル読み込み成功！")
        except Exception as e:
            st.error(f"エラー: {e}")

elif input_method == "Googleスプレッドシート":
    st.sidebar.info("事前にスプレッドシートの「共有」にサービスアカウントのアドレスを追加してください。")
    sheet_url = st.sidebar.text_input("スプレッドシートのURLを入力")
    
    if sheet_url:
        try:
            conn = st.connection("gsheets", type=GSheetsConnection)
            df = conn.read(spreadsheet=sheet_url, ttl=0)
            st.sidebar.success("スプレッドシート接続成功！")
        except Exception as e:
            st.sidebar.error(f"接続エラー: Secretsの設定またはURLを確認してください。\n{e}")

# --- 分析メイン処理 ---
if df is not None:
    df = df.dropna(how='all')

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 データ確認", 
        "📈 クロス集計", 
        "🚀 要因(ドライバー)分析", 
        "🌳 決定木分析", 
        "🧩 クラスター分析",
        "📖 分析手法の解説(用語集)",
        "🤖 AI分析アシスト"
    ])

    # --- タブ1: データ確認 ---
    with tab1:
        st.subheader("データプレビュー")
        st.dataframe(df)
        st.info(f"データサイズ: {df.shape[0]} 行, {df.shape[1]} 列")

    # --- タブ2: クロス集計 ---
    with tab2:
        st.subheader("クロス集計と可視化")
        
        col1, col2 = st.columns(2)
        with col1:
            index_col = st.selectbox("行（Index）を選択", df.columns, index=0, key="sb_cross_index")
        with col2:
            columns_col = st.selectbox("列（Column）を選択", df.columns, index=min(1, len(df.columns)-1), key="sb_cross_col")

        if index_col == columns_col:
            st.warning("⚠️ 行と列には異なる項目を選択してください。")
        else:
            df_cross = df.copy()
            df_cross[index_col] = enforce_likert_order(df_cross, index_col)
            df_cross[columns_col] = enforce_likert_order(df_cross, columns_col)
            
            calc_type = st.radio("集計値の表示形式", ["度数（人数）", "行パーセント（横計=100%）", "列パーセント（縦計=100%）"], horizontal=True)
            
            if calc_type == "度数（人数）":
                cross_tab = pd.crosstab(df_cross[index_col], df_cross[columns_col])
                fmt = ""
                val_name = "人数"
            elif calc_type == "行パーセント（横計=100%）":
                cross_tab = pd.crosstab(df_cross[index_col], df_cross[columns_col], normalize='index') * 100
                cross_tab = cross_tab.round(1)
                fmt = ".1f"
                val_name = "割合(%)"
            else:
                cross_tab = pd.crosstab(df_cross[index_col], df_cross[columns_col], normalize='columns') * 100
                cross_tab = cross_tab.round(1)
                fmt = ".1f"
                val_name = "割合(%)"

            st.write("##### 集計表")
            
            copy_text = cross_tab.to_csv(sep='\t')
            st_copy_to_clipboard(copy_text, "📋 表をコピー (ヘッダー付)", "✅ コピーしました！")
            
            st.dataframe(cross_tab) 

            graph_type = st.radio("グラフの種類", ["ヒートマップ", "積み上げ棒グラフ"], horizontal=True)
            if graph_type == "ヒートマップ":
                fig = px.imshow(cross_tab, text_auto=fmt if fmt else True, aspect="auto", color_continuous_scale='Blues')
            else:
                cross_tab_reset = cross_tab.reset_index().melt(id_vars=index_col, var_name=columns_col, value_name=val_name)
                fig = px.bar(cross_tab_reset, x=index_col, y=val_name, color=columns_col, title=f"{index_col} × {columns_col}", text_auto=fmt if fmt else True)
            st.plotly_chart(fig, use_container_width=True)

    # --- タブ3: 要因(ドライバー)分析 ---
    with tab3:
        st.subheader("🚀 要因（ドライバー）分析：オッズ比")
        
        st.info("""
        **💡 数値の見方（オッズ比）**
        * **1.0 より大きい**: その要因が結果を**促進**します。（例：2.0なら、その要因があると結果が2倍起こりやすい）
        * **1.0 より小さい**: その要因が結果を**抑制**します。（例：0.5なら、その要因があると結果が半分しか起こらない）
        """)

        col1, col2 = st.columns(2)
        with col1:
            target_col_reg = st.selectbox("目的変数（分析したい結果）", df.columns, index=0, key="sb_reg_target")
        with col2:
            valid_options_reg = [c for c in df.columns if c != target_col_reg]
            
            if "sb_reg_feature" in st.session_state:
                current_selection = st.session_state["sb_reg_feature"]
                safe_selection = [x for x in current_selection if x in valid_options_reg]
                st.session_state["sb_reg_feature"] = safe_selection

            default_feats_reg = [c for c in df.columns if c != df.columns[0]][:5]
            safe_default_reg = [x for x in default_feats_reg if x in valid_options_reg]

            feature_cols_reg = st.multiselect(
                "説明変数（背景・要因と思われる項目）", 
                valid_options_reg, 
                default=safe_default_reg, 
                key="sb_reg_feature"
            )

        if 'ai_msg_reg' in st.session_state and st.session_state['ai_msg_reg']:
            st.info("✅ AIがおすすめ設定を反映しました。下の「要因分析を実行」ボタンを押してください。")
            st.session_state['ai_msg_reg'] = False

        if st.button("要因分析を実行"):
            if not feature_cols_reg:
                st.warning("説明変数を1つ以上選択してください。")
            else:
                try:
                    df_reg = df[[target_col_reg] + feature_cols_reg].dropna()
                    
                    for col in df_reg.columns:
                        if df_reg[col].dtype == 'object':
                            le = LabelEncoder()
                            df_reg[col] = df_reg[col].astype(str)
                            df_reg[col] = le.fit_transform(df_reg[col])

                    X = df_reg[feature_cols_reg]
                    y = df_reg[target_col_reg]
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_scaled, y)

                    if model.coef_.shape[0] > 1: coefs = model.coef_[-1]
                    else: coefs = model.coef_[0]
                    
                    odds_ratios = np.exp(coefs)
                    res_df = pd.DataFrame({"要因": feature_cols_reg, "オッズ比": odds_ratios}).sort_values(by="オッズ比", ascending=True)

                    st.write(f"### 「{target_col_reg}」への影響度（オッズ比）")
                    fig = px.bar(res_df, x="オッズ比", y="要因", orientation='h', 
                                 title=f"「{target_col_reg}」に対するオッズ比（1.0が基準）",
                                 color="オッズ比", color_continuous_scale="RdBu_r", color_continuous_midpoint=1.0)
                    fig.add_vline(x=1.0, line_width=2, line_dash="dash", line_color="black")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("##### 詳細データ")
                    st_copy_to_clipboard(res_df.to_csv(sep='\t'), "📋 数値をコピー", "✅ コピーしました")
                    st.dataframe(res_df)

                except Exception as e:
                    st.error(f"分析エラー: {e}")

    # --- タブ4: 決定木分析 ---
    with tab4:
        st.subheader("決定木分析")
        st.caption("💡 図はマウスホイールで**拡大・縮小**、ドラッグで**移動**ができます。")
        
        col1, col2 = st.columns(2)
        with col1:
            target_col_tree = st.selectbox("目的変数（結果）", df.columns, index=0, key="sb_tree_target")
        with col2:
            valid_options = [c for c in df.columns if c != target_col_tree]
            
            if "sb_tree_feature" in st.session_state:
                current_selection = st.session_state["sb_tree_feature"]
                safe_selection = [x for x in current_selection if x in valid_options]
                st.session_state["sb_tree_feature"] = safe_selection

            default_feats = [c for c in df.columns if c != df.columns[0]][:3]
            safe_default = [x for x in default_feats if x in valid_options]

            feature_cols_tree = st.multiselect(
                "説明変数（要因）", 
                valid_options, 
                default=safe_default, 
                key="sb_tree_feature"
            )

        if 'ai_msg_tree' in st.session_state and st.session_state['ai_msg_tree']:
            st.info("✅ AIがおすすめ設定を反映しました。下の「決定木分析を実行」ボタンを押してください。")
            st.session_state['ai_msg_tree'] = False

        if st.button("決定木分析を実行"):
            if not feature_cols_tree:
                st.warning("説明変数を1つ以上選択してください。")
            else:
                try:
                    df_ml = df.copy()
                    
                    class_names_list = None
                    if df_ml[target_col_tree].dtype == 'object':
                        le_target = LabelEncoder()
                        df_ml[target_col_tree] = df_ml[target_col_tree].astype(str)
                        df_ml[target_col_tree] = le_target.fit_transform(df_ml[target_col_tree])
                        class_names_list = le_target.classes_.astype(str).tolist()
                    else:
                        class_names_list = sorted(df_ml[target_col_tree].unique().astype(str).tolist())

                    for col in feature_cols_tree:
                        if df_ml[col].dtype == 'object':
                            df_ml[col] = df_ml[col].astype(str)
                            le = LabelEncoder()
                            df_ml[col] = le.fit_transform(df_ml[col])

                    df_ml = df_ml.dropna(subset=[target_col_tree] + feature_cols_tree)
                    X = df_ml[feature_cols_tree]
                    y = df_ml[target_col_tree]

                    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
                    clf.fit(X, y)

                    tree_rules = export_text(clf, feature_names=feature_cols_tree)
                    st.write("##### 📋 分岐条件のテキスト詳細")
                    st_copy_to_clipboard(tree_rules, "📋 分岐ルールをコピー", "✅ コピーしました")
                    st.code(tree_rules)

                    dot_data = export_graphviz(
                        clf, out_file=None, feature_names=feature_cols_tree, class_names=class_names_list,
                        filled=True, rounded=True, special_characters=True, fontname="IPAexGothic"
                    )
                    st.graphviz_chart(dot_data)
                    
                    try:
                        graph = graphviz.Source(dot_data)
                        png_bytes = graph.pipe(format='png')
                        st.download_button("📥 決定木画像をダウンロード (PNG)", png_bytes, "decision_tree.png", "image/png")
                    except: pass

                    st.divider()
                    st.write("##### 📊 決定木データのダウンロード")
                    tree_df = get_decision_tree_data(clf, feature_cols_tree, class_names_list)
                    st.download_button("📥 決定木データをCSVでダウンロード", tree_df.to_csv(index=False).encode('utf-8_sig'), "decision_tree_data.csv", "text/csv")

                except Exception as e:
                    st.error(f"分析エラー: {e}")

    # --- タブ5: クラスター分析 ---
    with tab5:
        st.subheader("🧩 クラスター分析（セグメンテーション）")
        
        cluster_features = st.multiselect(
            "クラスター分析に使う変数を選択してください",
            df.columns,
            default=df.columns[:5].tolist(),
            key="sb_cluster_features"
        )
        
        n_clusters = st.slider("分類するグループ数", 2, 10, 4)

        if 'ai_msg_cluster' in st.session_state and st.session_state['ai_msg_cluster']:
            st.info("✅ AIがおすすめ設定を反映しました。下の「クラスター分析を実行」ボタンを押してください。")
            st.session_state['ai_msg_cluster'] = False

        if st.button("クラスター分析を実行"):
            if not cluster_features:
                st.warning("変数を選択してください")
            else:
                try:
                    df_cluster = df[cluster_features].dropna()
                    
                    for col in df_cluster.columns:
                        if df_cluster[col].dtype == 'object':
                            le = LabelEncoder()
                            df_cluster[col] = df_cluster[col].astype(str)
                            df_cluster[col] = le.fit_transform(df_cluster[col])

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(df_cluster)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(X_scaled)
                    
                    df['Cluster'] = clusters
                    df['Cluster_Name'] = df['Cluster'].apply(lambda x: f"グループ {x+1}")
                    st.success(f"{n_clusters}つのグループに分類しました！")

                    cluster_means_numeric = df_cluster.copy()
                    cluster_means_numeric['Cluster_Name'] = df['Cluster_Name']
                    cluster_means_numeric = cluster_means_numeric.groupby('Cluster_Name').mean()

                    st.write("##### グループごとの回答傾向（ヒートマップ）")
                    fig = px.imshow(cluster_means_numeric, text_auto=".2f", aspect="auto", color_continuous_scale="Viridis", title="クラスターごとの特徴比較")
                    st.plotly_chart(fig, use_container_width=True)

                    st.write("##### 分類結果付きデータのダウンロード")
                    st.download_button("📥 分類結果付きCSVをダウンロード", df.to_csv(index=False).encode('utf-8_sig'), 'clustered_data.csv', 'text/csv')
                    st.write("##### グループごとの人数")
                    st.dataframe(df['Cluster_Name'].value_counts().reset_index().rename(columns={'index':'Group', 'Cluster_Name':'Count'}))

                except Exception as e:
                    st.error(f"分析エラー: {e}")

    # --- タブ6: 解説 ---
    with tab6:
        st.header("📖 統計分析手法の解説ガイド")
        st.markdown("""
        このアプリで使用している分析手法について、「専門的な説明（Technical）」と「わかりやすい説明（Plain）」を併記しています。
        報告書作成やプレゼンテーションの際にご活用ください。
        """)
        
        st.divider()

        # 1. クロス集計
        st.subheader("1. クロス集計 (Cross Tabulation)")
        with st.expander("詳細を見る"):
            st.markdown("""
            #### 🛠 使用メソッド・原理
            * **手法**: 分割表 (Contingency Table) の作成
            * **統計的背景**: 2つの変数（質問項目）の間に「関連があるか」を見るために使用します。厳密には「カイ二乗検定 (Chi-square test)」を用いて、その偏りが偶然かどうかを判断することが一般的です。

            #### 💡 わかりやすい説明
            * **これ何？**: 「年代別 × 満足度」のように、2つの質問を掛け合わせて表にする最も基本的な分析です。
            * **目的**: 全体だけでは見えない、特定の属性（男女、年代など）ごとの違いを発見します。
            * **「有意差（意味のある差）」の目安**:
                * 一般的に、比較したいグループ間で **10%以上の差** があれば、「差がある」と見なして良いケースが多いです（マーケティング現場レベル）。
            """)

        # 2. ドライバー分析
        st.subheader("2. ドライバー分析 / 要因分析")
        with st.expander("詳細を見る"):
            st.markdown("""
            #### 🛠 使用メソッド・原理
            * **手法**: **ロジスティック回帰分析 (Logistic Regression)**
            * **アルゴリズム**: scikit-learnの `LogisticRegression` を使用。
            * **統計的背景**: 結果が「Yes/No（買った/買わない）」のような2値データの場合、通常の回帰分析は使えません。そこで「確率」を予測するロジスティック回帰を用います。
            * **係数の変換**: 算出された「偏回帰係数」を、指数変換（$e^x$）することで **「オッズ比 (Odds Ratio)」** に変換して表示しています。

            #### 💡 わかりやすい説明
            * **これ何？**: ある結果（例：商品を買った）に対して、どの要因がどれくらい影響したかを数値化する手法です。
            * **目的**: 施策の優先順位を決めるためです。「これを改善すれば、結果がこれだけ伸びる」というレバーを見つけます。
            * **「オッズ比」とは？**:
                * 結果の**「起こりやすさ」が何倍になるか**を表す数値です。
                * **1.0**: 影響なし（プラマイゼロ）。
                * **2.0**: その要素があると、結果が **2倍** 起こりやすくなる（強い促進要因）。
                * **0.5**: その要素があると、結果が **半分** しか起きなくなる（強い阻害要因）。
            """)

        # 3. 決定木分析
        st.subheader("3. 決定木分析 (Decision Tree)")
        with st.expander("詳細を見る"):
            st.markdown("""
            #### 🛠 使用メソッド・原理
            * **手法**: **CART法 (Classification and Regression Trees)**
            * **アルゴリズム**: scikit-learnの `DecisionTreeClassifier` を使用。
            * **統計的背景**: データを分割する際、**「Gini不純度 (Gini Impurity)」** という指標を使っています。これは「どれだけ綺麗にYes/Noが分かれたか」を計算するもので、この不純度が最も低くなる条件を探して自動的に分岐を作っています。

            #### 💡 わかりやすい説明
            * **これ何？**: 「もしAならB、そうでなければC」というように、結果に至る条件をツリー状に分解する手法です。
            * **目的**: 複雑な要因を整理し、**「一番影響力が大きい条件は何か？」**を視覚的に見つけるために使います。
            * **見方**:
                * **一番上の分岐**: これが**最も重要な要因**です。ここを見るだけで、結果を左右する最大のポイントがわかります。
            """)

        # 4. クラスター分析
        st.subheader("4. クラスター分析 (Cluster Analysis)")
        with st.expander("詳細を見る"):
            st.markdown("""
            #### 🛠 使用メソッド・原理
            * **手法**: **K-Means法 (K-平均法 / 非階層クラスター分析)**
            * **アルゴリズム**: scikit-learnの `KMeans` を使用。
            * **統計的背景**: データを $k$ 個のグループに分ける際、各グループの中心（重心）からの距離が最小になるように計算します。教師なし学習（正解データがいらない分析）の一種です。

            #### 💡 わかりやすい説明
            * **これ何？**: 回答パターンが似ている人を集めて、自動的にグループ（チーム）を作る手法です。
            * **目的**: 「セグメンテーション（顧客分類）」を行うためです。性別や年代だけでなく、「意識」や「行動」で分類することで、より刺さるメッセージを作れます。
            * **使い方のコツ**: 
                * 要因分析で「重要だ」とわかった項目を使ってクラスターを作ると、意味のあるグループができやすいです。
                * グループ名は、ヒートマップを見て人間が考えます（例：「価格重視派」「品質重視派」など）。
            """)

    # --- タブ7: AI分析アシスト ---
    with tab7:
        st.header("🤖 AI分析アシスト")
        
        st.subheader("Step 1: プロンプトをコピー")
        st.markdown("以下のプロンプトをコピーして、GeminiやChatGPTに**CSVファイルを添付**した状態で送信してください。")

        ai_prompt_text = """# 【絶対遵守のルール】
- ユーザーに「長くなるので省略します」「代表的なトップ3を挙げます」といった要約や省略を**絶対にしないでください**。
- 条件に合致する結果が見つかった場合、**見つかった数だけ、すべて漏らさず列挙**してください。

# 依頼
添付のアンケートデータを分析し、マーケティング戦略立案のための詳細なレポートを作成してください。
以下の4つの分析手法を用いて、トップライン（全体傾向）だけでなく、顕著な特徴や興味深い相関を「すべて網羅的に」抽出してください。

# 分析手法と観点
1. **クロス集計**
   - 年代、性別、居住年数などの基本属性と、満足度や意識設問との掛け合わせを分析してください。
   - 各意識・満足度のTop2（肯定派）の割合の差分が15%以上見られた組み合わせを**すべて**抽出してください。
   - さらに、Bottom2（否定派）の割合の差分が15%以上見られる組み合わせも**すべて**抽出してください。

2. **ドライバー分析（ロジスティック回帰）**
   - **⚠️重要: アンケート内にある「状態や意識を問う設問（例：愛着、住みやすさ、幸せ、良くしたい、評判が良い等）」を【すべて（最低でも5項目以上）】個別の目的変数として設定し、それぞれ回帰分析を行ってください。**
   - 説明変数：地域要素や満足度に関する設問（Q6系など）をすべて投入してください。
   - それぞれの目的変数に対し、P値が有意（< 0.05）なもののうち、オッズ比が1.0より大きくプラスの影響を与えている要素をランキング化してすべて列挙してください。
   - 同時に、オッズ比が0.8以下のマイナス（ネガティブ）の影響を与えている要素も分けてすべて列挙してください。

3. **決定木分析**
   - 目的変数：上記で設定した意識に関する重要指標（5項目以上）
   - どのような条件が重なると、その重要指標が高くなる（または低くなる）かの分岐ルールを見つけてください。

4. **クラスター分析**
   - 回答傾向が似ている回答者をグルーピング（3〜5グループ程度）してください。
   - 各グループの特徴（何に満足し、何に不満か）と、命名（ペルソナ名）を行ってください。

# 出力形式
1. **分析サマリー（人間が読む用）**
   - 各分析ごとに見出しを立て、箇条書きで詳細を記述してください。

2. **アプリ連携用設定データ（JSON形式）**
   - **最後に必ず**、以下のJSON形式で各分析ごとに設定パターンを抽出してください。
   - **⚠️【重要】「ドライバー分析」については、分析したすべての目的変数（最低でも5項目以上）をJSONのリストに含めてください。絶対に3つで省略しないでください。**
   - **列名は、CSVのヘッダーにある正確な名称を使用してください。**

```json
{
  "cross_tab": [
    {"index": "列名A", "columns": "列名B"},
    {"index": "列名C", "columns": "列名D"}
    // ... 見つけた数だけすべて記述する
  ],
  "driver_analysis": [
    {"target": "意識設問1", "features": ["説明変数1", "説明変数2", "説明変数3"]},
    {"target": "意識設問2", "features": ["説明変数1", "説明変数4"]},
    {"target": "意識設問3", "features": ["説明変数2", "説明変数5"]},
    {"target": "意識設問4", "features": ["説明変数1", "説明変数6"]},
    {"target": "意識設問5", "features": ["説明変数3", "説明変数7"]}
    // ... 分析した目的変数の数だけすべて（最低5つ以上）記述する
  ],
  "decision_tree": [
    {"target": "意識設問1", "features": ["説明変数1", "説明変数2", "説明変数3"]},
    {"target": "意識設問2", "features": ["説明変数1", "説明変数4", "説明変数5"]}
    // ...
  ],
  "clustering": [
    {"features": ["変数1", "変数2", "変数3"], "n_clusters": 4},
    {"features": ["変数1", "変数4", "変数5"], "n_clusters": 3}
  ]
}
```"""
        st_copy_to_clipboard(ai_prompt_text, "📋 プロンプトをコピー", "✅ コピーしました！")
        st.code(ai_prompt_text, language="markdown")
        
        st.divider()

        st.subheader("Step 2: AIの回答を貼り付け")
        st.markdown("AIが出力した**分析サマリー全体（最後のJSONまで含む）**をここに貼り付けてください。")

        ai_input = st.text_area("ここにGeminiの回答を貼り付けてください", height=300)

        if ai_input:
            json_match = re.search(r'```json\s*({.*?})\s*```', ai_input, re.DOTALL)
            
            config_str = None
            if json_match:
                config_str = json_match.group(1)
            else:
                fallback_match = re.search(r'\{.*"cross_tab".*\}', ai_input, re.DOTALL)
                if fallback_match:
                    config_str = fallback_match.group(0)

            if config_str:
                try:
                    config = json.loads(config_str)
                    st.success("✅ AIからの設定データを読み取りました！以下のおすすめパターンから選択してください。")
                    
                    col_ai_1, col_ai_2 = st.columns(2)
                    
                    if "cross_tab" in config and isinstance(config['cross_tab'], list):
                        with col_ai_1:
                            st.write(f"### 📈 クロス集計のおすすめ（{len(config['cross_tab'])}件）")
                            for i, setting in enumerate(config['cross_tab']):
                                with st.expander(f"パターン {i+1}: {setting.get('index')} × {setting.get('columns')}", expanded=False):
                                    st.write(f"**行**: {setting.get('index')}")
                                    st.write(f"**列**: {setting.get('columns')}")
                                    st.button(
                                        f"パターン{i+1}を適用",
                                        key=f"btn_cross_{i}",
                                        on_click=cb_update_cross,
                                        args=(setting.get('index'), setting.get('columns'))
                                    )

                    if "driver_analysis" in config and isinstance(config['driver_analysis'], list):
                        with col_ai_2:
                            st.write(f"### 🚀 ドライバー分析のおすすめ（{len(config['driver_analysis'])}件）")
                            for i, setting in enumerate(config['driver_analysis']):
                                tgt = setting.get('target')
                                feats = setting.get('features', [])
                                with st.expander(f"パターン {i+1}: {tgt}", expanded=False):
                                    st.write(f"**目的**: {tgt}")
                                    st.caption(f"**要因**: {', '.join(feats)}")
                                    valid_feats_reg = [f for f in feats if f in df.columns]
                                    st.button(
                                        f"パターン{i+1}を適用",
                                        key=f"btn_reg_{i}",
                                        on_click=cb_update_reg,
                                        args=(tgt, valid_feats_reg)
                                    )

                    col_ai_3, col_ai_4 = st.columns(2)

                    if "decision_tree" in config and isinstance(config['decision_tree'], list):
                        with col_ai_3:
                            st.write(f"### 🌳 決定木のおすすめ（{len(config['decision_tree'])}件）")
                            for i, setting in enumerate(config['decision_tree']):
                                tgt = setting.get('target')
                                feats = setting.get('features', [])
                                with st.expander(f"パターン {i+1}: {tgt}", expanded=False):
                                    st.write(f"**目的**: {tgt}")
                                    st.caption(f"**要因**: {', '.join(feats)}")
                                    valid_feats = [f for f in feats if f in df.columns]
                                    st.button(
                                        f"パターン{i+1}を適用",
                                        key=f"btn_tree_{i}",
                                        on_click=cb_update_tree,
                                        args=(tgt, valid_feats)
                                    )

                    if "clustering" in config and isinstance(config['clustering'], list):
                        with col_ai_4:
                            st.write(f"### 🧩 クラスター分析のおすすめ（{len(config['clustering'])}件）")
                            for i, setting in enumerate(config['clustering']):
                                feats = setting.get('features', [])
                                with st.expander(f"パターン {i+1}: {len(feats)}変数で分類", expanded=False):
                                    st.caption(f"**変数**: {', '.join(feats)}")
                                    valid_feats_cluster = [f for f in feats if f in df.columns]
                                    st.button(
                                        f"パターン{i+1}を適用",
                                        key=f"btn_cluster_{i}",
                                        on_click=cb_update_cluster,
                                        args=(valid_feats_cluster,)
                                    )

                except json.JSONDecodeError as e:
                    st.error(f"JSONの解析に失敗しました。AIの出力が正しいフォーマットか確認してください。\n詳細エラー: {e}")
            else:
                st.warning("設定データ(JSON)が見つかりません。プロンプトの指示通りにGeminiが出力しているか確認してください。")

else:
    st.info("👈 左側のサイドバーからデータを選択してください。")
