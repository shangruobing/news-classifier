from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def fit_model(training_data, target):
    """填充并训练模型"""
    x_train, x_test, y_train, y_test = train_test_split(training_data,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=29)
    pipe = Pipeline(
        [('vet', CountVectorizer()), ('tfidf', TfidfTransformer()),
         ('model', LogisticRegression())])
    model = pipe.fit(x_train, y_train)

    prediction = model.predict(x_test)
    print("accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100, 2)))

    return model


def model_predict(model, test_data):
    """使用模型来预测数据"""
    count = 0
    prediction = []
    true_number = 0
    fake_number = 0
    error_number = 0
    while len(test_data) > count:
        predict = model.predict(test_data.iloc[count])
        prediction.append(''.join(predict).upper())

        if predict == "true":
            true_number += 1
        elif predict == "fake":
            fake_number += 1
        else:
            error_number += 1
        count += 1

    print(f"新闻总数：{true_number + fake_number}")
    print(f"真新闻数量：{true_number}")
    print(f"假新闻数量：{fake_number}")
    print(f"错误记录数量：{error_number}")

    return prediction
