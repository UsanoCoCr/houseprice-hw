full.drop(['ADDRESS'], axis=1, inplace=True)
full.drop(['APARTMENT NUMBER'], axis=1, inplace=True)
full.drop(['SALE DATE'], axis=1, inplace=True)

full['LAND SQUARE FEET'] = full['LAND SQUARE FEET'].transform(lambda x: 0 if x == ' -  ' else float(x))
full['GROSS SQUARE FEET'] = full['GROSS SQUARE FEET'].apply(lambda x: 0 if x == ' -  ' else float(x))

verNei = full.groupby(['NEIGHBORHOOD'])[['SALE PRICE']].agg(['mean'])
dNei = dict(dict(verNei[('SALE PRICE', 'mean')]))
verBOR = full.groupby(['BOROUGH'])[['SALE PRICE']].agg(['mean'])
dBOR = dict(verBOR[('SALE PRICE', 'mean')])
verBCP = full.groupby(['BUILDING CLASS AT PRESENT'])[['SALE PRICE']].agg(['mean'])
dBCP = dict(verBCP[('SALE PRICE', 'mean')])
verBCA = full.groupby(['BUILDING CLASS AT TIME OF SALE'])[['SALE PRICE']].agg(['mean'])
dBCA = dict(verBCA[('SALE PRICE', 'mean')])
verTCP = full.groupby(['TAX CLASS AT PRESENT'])[['SALE PRICE']].agg(['mean'])
dTCP = dict(verTCP[('SALE PRICE', 'mean')])
verTCA = full.groupby(['TAX CLASS AT TIME OF SALE'])[['SALE PRICE']].agg(['mean'])
dTCA = dict(verTCA[('SALE PRICE', 'mean')])

full["oNeighborhood"] = full["NEIGHBORHOOD"].map(dNei).fillna(1).apply(np.log)
full["oBOR"] = full["BOROUGH"].map(dBOR).fillna(1).apply(np.log)
full["oBCA"] = full["BUILDING CLASS AT TIME OF SALE"].map(dBCA).fillna(1).apply(np.log)
full["oBCP"] = full["BUILDING CLASS AT PRESENT"].map(dBCP).fillna(1).apply(np.log)
full["oTCA"] = full["BUILDING CLASS AT TIME OF SALE"].map(dTCA).fillna(1).apply(np.log)
full["oTCP"] = full["BUILDING CLASS AT PRESENT"].map(dTCP).fillna(1).apply(np.log)
# full["oBCC"] = full["BUILDING CLASS CATEGORY"].map(dTCP).fillna(1).apply(np.log)

full.drop(['BOROUGH'], axis=1, inplace=True)
full.drop(['BUILDING CLASS AT TIME OF SALE'], axis=1, inplace=True)
full.drop(['BUILDING CLASS AT PRESENT'], axis=1, inplace=True)
full.drop(['NEIGHBORHOOD'], axis=1, inplace=True)


NumStr = [
    "YEAR BUILT",
    "TAX CLASS AT PRESENT",
    "TAX CLASS AT TIME OF SALE",
    "BUILDING CLASS CATEGORY",
    ]
for col in NumStr:
    full[col] = full[col].astype(str)

full.drop(['SALE PRICE'], axis=1, inplace=True)


class labelenc(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        lab = LabelEncoder()
        X["YEAR BUILT"] = lab.fit_transform(X["YEAR BUILT"])
        return X

class skew_dummies(BaseEstimator, TransformerMixin):
    def __init__(self, skew=0.5):
        self.skew = skew

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_numeric = X.select_dtypes(exclude=["object"])
        skewness = X_numeric.apply(lambda x: skew(x))

        skewness_features = skewness[abs(skewness) >= self.skew].index
        X[skewness_features] = np.log1p(X[skewness_features])
        X = pd.get_dummies(X)
        return X

# build pipeline
pipe = Pipeline([
    ('labenc', labelenc()),
    ('skew_dummies', skew_dummies(skew=1)),
])


full2 = full.copy()
data_pipe = pipe.fit_transform(full2)

scaler = RobustScaler()  # 对数据进行缩放

n_train = train.shape[0]  # 获得有多少条训练数据

X = data_pipe[:n_train]  # 获得训练数据
test_X = data_pipe[n_train:]  # 获得测试数据
y = train_y
X_scaled = scaler.fit(X).transform(X)
y_log = np.log(y)  # 对于 y 进行人为定义log 进行缩放
test_X_scaled = scaler.transform(test_X)  # 对测试数据进行缩放

# ## Feature Selection

# + __上面的特征工程还不够，所以我们需要更多.__   
# + __组合不同的特征通常是一个好方法，但我们不知道应该选择什么特征。 幸运的是，有些模型可以提供特征选择，这里我使用 Lasso，但你可以自由选择 Ridge、RandomForest 或 GradientBoostingTree.__

lasso = Lasso(alpha=0.001)
lasso.fit(X_scaled, y_log)  # 用lasso 拟合 缩放后的训练数据 和 对应的 y 值

# In[ ]:


FI_lasso = pd.DataFrame({"Feature Importance": lasso.coef_}, index=data_pipe.columns)

# In[ ]:


FI_lasso.sort_values("Feature Importance", ascending=False)  # 利用Lasso计算出一个特征重要性的分数

# In[ ]:


FI_lasso[FI_lasso["Feature Importance"] != 0].sort_values("Feature Importance").plot(kind="barh", figsize=(15, 25))
plt.xticks(rotation=90)
plt.show()  ### 更直观的可视化一下

class AverageWeight(BaseEstimator, RegressorMixin):
    def __init__(self, mod, weight):
        self.mod = mod
        self.weight = weight

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.mod]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        w = list()
        pred = np.array([model.predict(X) for model in self.models_])
        # for every data point, single model prediction times weight, then add them together
        for data in range(pred.shape[1]):
            single = [pred[model, data] * weight for model, weight in zip(range(pred.shape[0]), self.weight)]
            w.append(np.sum(single))
        return w


weight_avg = AverageWeight(mod=[RandomForestRegressor(), XGBRegressor(), ExtraTreesRegressor()], weight=[0.4,0.4,0.2])

weight_avg.fit(X_scaled, y_log)
y = weight_avg.predict(test_X_scaled)
y_pred = np.exp(y)