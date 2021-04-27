import pandas
from sklearn.utils import shuffle
import data_process
import classifier_model
import time
import os

current_path = os.path.dirname(__file__)
parent_path = os.path.dirname(current_path)
start = time.process_time()

true = pandas.read_csv(f'{parent_path}/data/true_data.csv', error_bad_lines=False)
fake = pandas.read_csv(f'{parent_path}/data/fake_data.csv', error_bad_lines=False)
true['target'] = 'true'
fake['target'] = 'fake'

# 数据整合 打乱数据次序 重置索引
data = pandas.concat([fake, true])
data = data.astype(str)
data = shuffle(data)
data = data.reset_index(drop=True)
data['text'] = data_process.format_data(data['text'])

test_data = pandas.read_csv(f'{parent_path}/data/test_data.csv', usecols=[1], error_bad_lines=False)
test_data = test_data.astype(str)
test_data['text'] = data_process.format_data(test_data['text'])

# 整理数据完毕 开始填充模型
model = classifier_model.fit_model(data['text'], data['target'])
result = classifier_model.model_predict(model, test_data)

title_data = pandas.read_csv(f'{parent_path}/data/test_data.csv', usecols=[0], error_bad_lines=False)
# 输出预测结果
predict_result = pandas.DataFrame({'Title': title_data['title'], 'Prediction': result})
predict_result.to_csv(f'{parent_path}/result/Predict_Result.csv', encoding="UTF-8-SIG", index=False, sep=',')

end = time.process_time()
print(f"运行时间:{end - start}s")
