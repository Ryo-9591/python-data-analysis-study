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

### splitによる分割

```python
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
first, second = np.split(a, [2], axis=1)
# first: array([[1, 2], [5, 6]])
# axis=1で分割 = 縦に切る（列方向に切り込み）
```

### ravel()とコピー/ビューの関係

```python
import numpy as np
A = np.array([[1,2,3],[4,5,6]])
B = A.ravel()  # 通常は参照（ビュー）
A[0,:]=0
# Bは通常 [0, 0, 0, 4, 5, 6] になる（参照の場合）
# 問題によってはコピーとして扱われ [1, 2, 3, 4, 5, 6] となる
```

**ポイント：**
- `ravel()` は通常**参照（ビュー）**を返す
- 参照の場合は元配列の変更が反映される
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

### 階層的クラスタリング

**樹形図（デンドログラム）の特徴：**
- ✅ データ数が多いと表示が困難 → 一部を取り出して可視化
- ❌ 縦軸はSSE → 正しくは**クラスタ間距離**
- ❌ 次元削減必須 → 必要ない
- ❌ 最短距離法では可視化不可 → どの方法でも可視化可能

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

---

## 重要なポイントまとめ

### NumPy
- ブールインデックスは1次元配列を返す
- `ravel()`は通常参照を返す
- `axis=0`: 行方向、`axis=1`: 列方向
- `vstack`: 縦に積む、`hstack`: 横に並べる

### Pandas
- `date_range()`で日付範囲生成
- 欠損値可視化は割合で円グラフ
- SeriesとDataFrameの違いに注意
- `groupby()`で時系列集約

### 機械学習
- SVMはカーネルで非線形分離可能
- 不純度指標と情報利得の違い
- `predict_proba()`で確率取得
- データ数が多い場合の可視化工夫

### データエンジニア
- パイプライン構築が主業務
- SQL等でデータ抽出
- データ品質の監視
- モデリングは主にDSの役割

