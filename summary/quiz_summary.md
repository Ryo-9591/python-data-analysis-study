# データ分析学習まとめ

## 目次
- [NumPy](#numpy)
- [Pandas](#pandas)
- [機械学習](#機械学習)
- [データエンジニアリング](#データエンジニアリング)
- [その他](#その他)

---

## NumPy

### ブールインデックス抽出

```python
import numpy as np
A = np.array([[0, 1, 2, 3, 4]])
B = np.full((1, 5), 1)
result = A[A>=B]
# 結果: array([1, 2, 3, 4])
```

**ポイント：**
- ブールインデックス配列 `A>=B` の形状は `(1, 5)` の2次元配列
- ブールインデックスで抽出すると結果は**1次元配列**になる

### 行列の掛け算

```
(2) × (2) は計算できない
(2)   (2)
```

**理由：** Aの列数(1) ≠ Bの行数(2) のため、行列の積は定義されない。

### vsplitによる分割と転置

```python
import numpy as np
A = np.eye(4)
first, second = np.vsplit(A, [3])
result = second.T
# 結果: array([[0.], [0.], [0.], [1.]])
```

**ステップ解説：**
1. `np.eye(4)` で4×4の単位行列を作成
2. `np.vsplit(A, [3])` で3行目以降を `second` に分割
3. `second.T` で転置 → 4×1の配列

### vstack/vsplitとaxisの関係

```
# vstackは縦に積む (axis=0)
# vsplitは縦に分割する (axis=0)
# axis=1で分割すると縦に割られる (列方向の分割)
```

**axisの覚え方：**
- **v**stack = **v**ertical = **縦方向** = axis=0（行の操作）
- **h**stack = **h**orizontal = **横方向** = axis=1（列の操作）
- splitは軸方向に**切り込みを入れる**操作

### axis=0で縦に積む

```python
import numpy as np

A = np.array([[1, 2, 3, 4]])
B = np.array([[5, 6, 7, 8],
              [9, 10, 11, 12]])

# axis=0 (行方向) に結合 = 縦に積む
C = np.concatenate((A, B), axis=0)
# 結果:
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
```

### 配列の結合と出力

```python
import numpy as np

A = np.full((2, 2), 1)  # 2×2で全て1の配列
B = np.zeros((2, 2))     # 2×2で全て0の配列

# axis=0で結合（縦に積む）
result = np.concatenate([A, B], axis=0)
# 結果:
# array([[1., 1.],
#        [1., 1.],
#        [0., 0.],
#        [0., 0.]])
```

**ポイント：**
- `np.full(shape, value)`: 指定形状で指定値で埋める
- `np.zeros(shape)`: 指定形状で0で埋める
- 異なるデータ型を結合すると、広い型（ここでは浮動小数点）に統一される

### splitによる分割

```python
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
first, second = np.split(a, [2], axis=1)
# first: array([[1, 2], [5, 6]])
# axis=1で分割 = 縦に切る（列方向に切り込み）
```

### hsplitによる水平分割

```python
import numpy as np

A = np.eye(4)  # 4×4の単位行列
first, second = np.hsplit(A, [2])  # インデックス2の前で分割

# first: 最初の2列（インデックス0, 1）
# 結果:
# array([[1., 0.],
#        [0., 1.],
#        [0., 0.],
#        [0., 0.]])

# second: 残りの2列（インデックス2, 3）
```

**ポイント：**
- `np.hsplit()`: 水平方向（列方向）に分割
- `np.vsplit()`: 垂直方向（行方向）に分割
- `np.split(..., axis=1)`: `hsplit()` と同等
- `np.split(..., axis=0)`: `vsplit()` と同等

### ravel()とflatten()の違い

```python
import numpy as np
A = np.array([[1, 2, 3], [4, 5, 6]])

# ravel(): 通常は参照（ビュー）を返す
B = A.ravel()
A[0, :] = 0
# Bは通常 [0, 0, 0, 4, 5, 6] になる（参照の場合）

# flatten(): 必ずコピーを返す
C = A.flatten()
A[1, :] = 0
# Cは [1, 2, 3, 4, 5, 6] のまま（コピーのため）
# C[-1] = 6 が返される
```

**ポイント：**
- `ravel()`: 通常**参照（ビュー）**を返す → 元配列の変更が反映される
- `flatten()`: **必ずコピー**を返す → 元配列の変更が反映されない
- 確実にコピーしたい場合は `flatten()` を使用

### meshgridとグリッド生成

```python
m = np.arange(0, 4)
n = np.arange(4, 8)
xx, yy = np.meshgrid(m, n)
```

### ユニバーサル関数

```python
b = np.arange(-3, 3).reshape((2, 3))
np.abs(b)     # 絶対値
np.sin(b)     # 三角関数
np.log(b)     # 自然対数
np.log10(b)   # 常用対数
np.exp(b)     # 指数関数
```

### 乱数生成（np.random.rand）

```python
import numpy as np

# [0.0, 1.0) の範囲の一様分布から乱数を生成
A = np.random.rand(10, 1)  # 10行1列の配列
```

**ポイント：**
- `np.random.rand(shape)`: **一様分布**から [0.0, 1.0) の範囲で乱数生成
- 形状は `(10, 1)` などのタプルで指定
- ❌ 正規分布ではない → 正規分布には `np.random.randn()` や `np.random.normal()` を使用

**違い：**
- `rand()`: 一様分布 [0.0, 1.0)
- `randn()`: 標準正規分布（平均0、標準偏差1）
- `normal(loc, scale, size)`: 任意の正規分布

---

## Pandas

### 日付範囲の生成

```python
pd.date_range(start="2020-01-01", end="2020-1-31")
# 結果: 2020年1月1日から31日までのDatetimeIndex
```

**間違いやすいポイント：**
- `pd.date()` は存在しない
- `pd.date_range()` が正しい

### 欠損値の割合を円グラフで可視化

```python
import matplotlib.pyplot as plt

plt.pie(df.isnull().sum()/len(df), 
        labels=df.columns, 
        autopct="%1.1f%%")
plt.show()
```

**正しいコード：**
- `df.isnull().sum()` で欠損値の個数
- `/len(df)` で割合に変換
- 円グラフの引数は**割合のリスト**

### カラムの抽出（Series）

```python
# 単一カラムをSeriesとして抽出
df["name of country"]

# インデックス指定でSeries
df.iloc[:, 0]  # 最初の列をSeriesとして取得

# 注意: 以下の方法はDataFrameを返す
df[["name of country"]]  # DataFrame
df.loc[:, ["name of country"]]  # DataFrame
df.filter(["name of country"])  # DataFrame
```

### グループ化と集計

```python
# 水曜日までの1週間単位の合計
df.groupby(pd.Grouper(freq="W-WED")).sum()

# 月ごとの平均
df.groupby(pd.Grouper(freq="ME")).mean()

# resampleで月ごと
df["カラム名"].resample("ME").mean()
```

**頻度指定：**
- `W-WED`: 水曜日を週末とする週単位
- `ME`: Month End（月末）

### DataFrameへの列追加

```python
import pandas as pd

# 方法1: 直接代入（推奨）
df["増減値"] = df["終値"] - df["始値"]

# 方法2: locを使用
df.loc[:, "増減値"] = df.loc[:, "終値"] - df.loc[:, "始値"]

# 方法3: ilocを使用（インデックス指定）
df.loc[:, "増減値"] = df.iloc[:, 2] - df.iloc[:, 1]

# ❌ 間違い: リスト同士の引き算はできない
# df.loc[:, "増減値"] = df.loc[["終値"] - ["始値"]]  # エラー
```

### 複数条件でのフィルタリング

```python
# AND条件（両方の条件を満たす）
df[(df["A"] == 3000) & (df["B"] == 3000)]

# OR条件（どちらかの条件を満たす）
df[(df["A"] == 3000) | (df["B"] == 3000)]
```

**重要：**
- Pythonの `and`, `or` は使えない
- Pandas/NumPyでは `&` (AND), `|` (OR), `~` (NOT) を使用
- 条件は括弧 `()` で囲む必要がある

### 時系列データの可視化

```python
import matplotlib.pyplot as plt

# 日付ごとの推移を折れ線グラフで表示
df["利用回数"].plot()
plt.show()
```

**ポイント：**
- Pandas Seriesの `.plot()` はデフォルトで折れ線グラフ
- Seriesのインデックス（日付）が自動的にX軸になる
- 時系列データの推移には折れ線グラフが適している

### datetime型への変換

```python
# オブジェクト型（文字列）をdatetime型に変換
df["Date"] = pd.to_datetime(df["Date"])
```

**間違いやすいポイント：**
- ❌ `pd.datetime()` は存在しない
- ❌ `pd.to_date()` は存在しない（正しくは `pd.to_datetime()`）
- ❌ `astype(datetime)` は推奨されない（形式が複雑な場合は解析できない）
- ✅ `pd.to_datetime()` が最も柔軟で推奨される方法

---

## 機械学習

### サポートベクターマシン（SVM）

**正しい記述：**
- scikit-learnでカーネルはガウスカーネル以外にも指定可能
  - 線形カーネル: `'linear'`
  - 多項式カーネル: `'poly'`
  - シグモイドカーネル: `'sigmoid'`

**間違いやすいポイント：**
- ❌ SVMは欠損値を内部で処理できない → 前処理が必要
- ❌ 線形分離可能な問題にのみ適用 → カーネルトリックで非線形分離可能
- ❌ マージン距離を最小化 → 正しくは**最大化**

### ロジスティック回帰

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# 予測確率を取得
proba = model.predict_proba(X_test)  # 各クラスの確率

# 予測ラベル
labels = model.predict(X_test)  # 最も確率の高いクラス

# 対数確率
log_proba = model.predict_log_proba(X_test)
```

**メソッドの違い：**
- `predict_proba()`: 確率を返す
- `predict()`: ラベルを返す
- `predict_log_proba()`: 対数確率を返す
- `get_params()`: ハイパーパラメータを返す

### 決定木の不純度指標

**不純度の指標：**
- ✅ エントロピー (Entropy)
- ✅ ジニ不純度 (Gini Impurity) - scikit-learnのデフォルト
- ✅ 分類誤差 (Classification Error)

**間違いやすいポイント：**
- ❌ 情報利得 (Information Gain) → これは不純度ではなく、不純度の減少量

**注：** 情報利得 = 分割前の不純度 - 分割後の不純度

### DecisionTreeClassifierの引数

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=5,           # 木の深さ（指定可能）
    max_leaf_nodes=10,     # 葉の数（指定可能）
    criterion='gini'      # 不純度の指標（指定可能）
)
```

**指定可能な引数：**
- ✅ `max_depth`: 木の深さ
- ✅ `max_leaf_nodes`: 葉の数
- ✅ `criterion`: 不純度の指標（'gini', 'entropy', 'log_loss'）

**指定できない引数：**
- ❌ `n_estimators`（木の数）: DecisionTreeClassifierは**単一の木**を構築するため
  - 複数の木を使う場合は `RandomForestClassifier` や `GradientBoostingClassifier` を使用

### 階層的クラスタリング

**樹形図（デンドログラム）の特徴：**
- ✅ データ数が多いと表示が困難 → 一部を取り出して可視化
- ❌ 縦軸はSSE → 正しくは**クラスタ間距離**
- ❌ 次元削減必須 → 必要ない
- ❌ 最短距離法では可視化不可 → どの方法でも可視化可能

### Ridge回帰とLasso回帰の正則化項

| 回帰手法 | 正則化項 | 特徴 |
|---------|---------|------|
| **Ridge回帰** | **L2正則化項** | 重みをゼロに近づけるが、完全にゼロにはしない<br>重みの二乗和のペナルティ |
| **Lasso回帰** | **L1正則化項** | 不要な特徴量の重みを完全にゼロにする（スパース性）<br>重みの絶対値の和のペナルティ |

**正則化項の数式：**
- L2正則化項: λ∑βⱼ²（重みの二乗和）
- L1正則化項: λ∑|βⱼ|（重みの絶対値の和）

**用途：**
- Ridge回帰: 過学習を防ぎ、モデルを安定させる
- Lasso回帰: 特徴量選択（変数選択）を行い、解釈性を向上させる

### ROC曲線

**ROC曲線（Receiver Operating Characteristic Curve）の軸：**

| 軸 | 指標 | 定義 |
|----|------|------|
| **横軸（X軸）** | **偽陽性率 (FPR)** | FPR = FP / (FP + TN)<br>実際は陰性なのに陽性と予測された割合 |
| **縦軸（Y軸）** | **真陽性率 (TPR)** | TPR = TP / (TP + FN)<br>実際は陽性で正しく陽性と予測された割合<br>（= 再現率 Recall） |

**補足：**
- PR曲線（Precision-Recall Curve）の軸は「再現率（横軸）と適合率（縦軸）」
- ROC曲線は閾値を変化させた時の分類性能を可視化

### 欠損値の対処方法

**一般的な対処方法：**
- ✅ 統計的な代表値で埋める（平均値、中央値、最頻値）
- ✅ 欠損値があるデータを用いない（完全ケース分析）
- ✅ 欠損値に対応するアルゴリズムを選定（XGBoost、LightGBMなど）

**間違いやすいポイント：**
- ❌ One-hotエンコーディング → これは**カテゴリ変数の数値化**手法であり、欠損値の補完方法ではない

**注意：**
- 欠損値の対処方法は、データの性質、欠損のパターン、情報損失とのトレードオフを考慮して選択する

---

## データエンジニアリング

### データエンジニアの業務

**正しい業務：**
- ✅ 集計ミスがないかの確認
- ✅ データサイエンティストや顧客とのコミュニケーション
- ✅ データベース言語によるデータ抽出

**誤りやすいポイント：**
- ❌ 機械学習アルゴリズムの深い理解 → これはデータサイエンティストの役割

**役割の違い：**
- **データエンジニア**: データパイプライン、インフラ構築、ETL/ELT
- **データサイエンティスト**: モデリング、アルゴリズム選択、分析

---

## その他

### Python基礎

#### リストの繰り返し

```python
A = [1] * 10
# 結果: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

**ポイント：** `*` 演算子はリストの要素を繰り返す

#### pip freeze vs pip list

| コマンド | 用途 | 出力形式 |
|---------|------|---------|
| `pip freeze` | 環境再現用 | `name==version` |
| `pip list` | 一覧確認用 | 表形式 |

```bash
# requirements.txtの作成
pip freeze > package_list.txt
```

#### 仮想環境から抜ける

```bash
deactivate  # venvやAnaconda環境から抜ける
```

**注意：**
- `exit`: シェルセッション終了
- `quit`: Pythonインタープリタ終了
- `escape`: 関係ない

### Matplotlib

#### Jupyter Notebookのマジックコマンド

**正しい記述：**
- `%timeit`: 一行のコードの実行時間を計測
- `%%timeit`: セル全体のコードの実行時間を計測
- `%lsmagic`: マジックコマンド一覧を表示

**間違った記述：**
- ❌ `%matplotlib tk` はセル直下にグラフ出力 → 正しくは独立ウィンドウ表示
- ✅ セル直下に出力するのは `%matplotlib inline`

**バックエンドの違い：**
- `%matplotlib inline`: セル直下に出力（デフォルト）
- `%matplotlib tk`: 独立ウィンドウで表示
- `%matplotlib notebook`: インタラクティブ表示

#### hist()メソッドの引数

```python
import matplotlib.pyplot as plt

plt.hist(data, bins=20)                    # ビンの数を指定
plt.hist(data, bins='auto')                 # 最適なビンの数を自動決定
plt.hist([data1, data2], stacked=True)      # 積み上げヒストグラム
plt.hist(data, density=True)                # 確率密度分布（面積=1）
```

**引数の機能：**
- ✅ `bins`: ビンの数を変更（整数、配列、または'auto', 'sturges', 'fd', 'scott'などの文字列）
- ✅ `stacked=True`: 積み上げヒストグラムを描画
- ✅ `density=True`: 確率密度分布を表示（面積の合計が1）

**間違いやすいポイント：**
- ❌ 「相対度数分布を表示できる」→ `density=True` は確率密度分布であり、相対度数分布とは異なる
  - 相対度数分布: バーの高さの合計が1
  - 確率密度分布: バーの面積の合計が1（ビンの幅が均一でない場合は高さの合計≠1）

### 数学・統計

#### 対数の計算

```
log 2 + log 5 = log (2 × 5) = log 10
```

**対数の和の公式：**
```
log_a M + log_a N = log_a (M × N)
```

#### 相関係数

- **ピアソンの相関係数**: pandas `corr()` のデフォルト
- スピアマン: `df.corr(method='spearman')`
- ケンドール: `df.corr(method='kendall')`

#### ポアソン分布

**ポアソン分布の特徴：**
- ✅ 離散型確率分布
- ✅ 稀な事象の発生回数をモデル化するのに適している
  - 例: 交通事故の発生回数、機械部品の故障予測、電話の着信回数
- ✅ パラメーターは平均値 λ（ラムダ）の1つだけ
  - 平均値 = 分散 = λ
  - 標準偏差 = √λ

**間違いやすいポイント：**
- ❌ パラメーターは平均値と標準偏差 → パラメーターは平均値λだけ
- ❌ 連続型確率分布 → 離散型確率分布（回数や個数を扱うため）
- ❌ 試行回数と平均値が∞に発散する極限 → 正しくは試行回数n→∞、成功確率p→0、積np→λ（有限）の極限

#### 集合の演算

```
A = {1,2,3,4}
B = {3,4,5,6}

A ∪ B = {1,2,3,4,5,6}  # 和集合
A ∩ B = {3,4}           # 積集合（共通部分）
```

### datetimeモジュール

**正しい記述：**
- ✅ timezoneを指定できる
- ✅ `now()` で現在の日時を取得
- ✅ 日時⇔文字列の変換が可能

**間違った記述：**
- ❌ 日付同士の演算ができない → `timedelta`で計算可能

```python
from datetime import datetime, timedelta

# 日数の差分計算
date1 = datetime(2024, 1, 1)
date2 = datetime(2024, 1, 31)
diff = date2 - date1
print(diff.days)  # 30

# 日時→文字列
date1.strftime("%Y-%m-%d")

# 文字列→日時
datetime.strptime("2024-01-01", "%Y-%m-%d")
```

### pandasインデックス操作

```
df.iloc[1:]  # 1行目から最後まで（1行目を含む）
df.iloc[:1]  # 最初から1行目まで（1行目を含む）
```

**スライス記法：**
- `[start:end]`: startは含む、endは含まない
- `[1:]`: インデックス1から最後まで
- `[:1]`: 最初からインデックス1の前まで

### Seriesの条件判定

```python
import pandas as pd

df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})

# 特定の行を抽出して条件判定
result = df.iloc[1] > 3
# 結果: Series
# A    False
# B     True
# dtype: bool
```

**ポイント：**
- `df.iloc[行番号]` はSeriesを返す
- Seriesに条件演算子を適用すると、各要素に対して判定が行われ、ブール値のSeriesが返される

---

## 重要なポイントまとめ

### NumPy
- ブールインデックスは1次元配列を返す
- `ravel()`は通常参照を返す、`flatten()`は必ずコピー
- `axis=0`: 行方向、`axis=1`: 列方向
- `vstack`: 縦に積む、`hstack`: 横に並べる
- `np.random.rand()`: 一様分布、正規分布ではない

### Pandas
- `date_range()`で日付範囲生成
- `pd.to_datetime()`でdatetime型に変換
- 欠損値可視化は割合で円グラフ
- SeriesとDataFrameの違いに注意
- 複数条件のフィルタリングは `&`, `|` を使用（`and`, `or` は不可）
- `groupby()`で時系列集約
- Seriesの `.plot()` は時系列可視化に便利

### 機械学習
- SVMはカーネルで非線形分離可能
- 不純度指標と情報利得の違い
- `predict_proba()`で確率取得
- DecisionTreeClassifierは単一の木（木の数を指定する引数なし）
- Ridge回帰: L2正則化、Lasso回帰: L1正則化
- ROC曲線: 横軸=FPR、縦軸=TPR
- 欠損値対処: One-hotエンコーディングは使わない（カテゴリ変数用）

### 数学・統計
- ポアソン分布: 離散型、パラメーターはλのみ、稀な事象のモデル化

### データエンジニア
- パイプライン構築が主業務
- SQL等でデータ抽出
- データ品質の監視
- モデリングは主にDSの役割

