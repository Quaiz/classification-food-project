import os
import matplotlib.pyplot as plt
import seaborn as sns

# Đường dẫn dataset
train_data = "D:\\ThirdYearsInHell\\deeplearning\\Project\\DatasetImageFoodVietNam1111\\DatasetImageFoodVietNam1111\\Train"
test_data = "D:\\ThirdYearsInHell\\deeplearning\\Project\\DatasetImageFoodVietNam1111\\DatasetImageFoodVietNam1111\\Test"
validate_data = "D:\\ThirdYearsInHell\\deeplearning\\Project\\DatasetImageFoodVietNam1111\\DatasetImageFoodVietNam1111\\Validate"

# Hàm đếm số ảnh
def count_images_in_folders(directory):
    categories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    counts = [len(os.listdir(os.path.join(directory, c))) for c in categories]
    return categories, counts

# Thống kê và vẽ biểu đồ
categories, counts_train = count_images_in_folders(train_data)
train_data_dict = dict(zip(categories, counts_train))
print("Train data:", train_data_dict)

plt.figure(figsize=(8,5))
sns.barplot(x=list(train_data_dict.keys()), y=list(train_data_dict.values()))
plt.xlabel("Loại")
plt.ylabel("Số lượng hình ảnh")
plt.title("Số lượng hình ảnh của tập Train")
plt.show()

categories, counts_validate = count_images_in_folders(validate_data)
validate_data_dict = dict(zip(categories, counts_validate))
print("Validate data:", validate_data_dict)

plt.figure(figsize=(8,5))
sns.barplot(x=list(validate_data_dict.keys()), y=list(validate_data_dict.values()))
plt.xlabel("Loại")
plt.ylabel("Số lượng hình ảnh")
plt.title("Số lượng hình ảnh của tập Validate")
plt.show()

categories, counts_test = count_images_in_folders(test_data)
test_data_dict = dict(zip(categories, counts_test))
print("Test data:", test_data_dict)

plt.figure(figsize=(8,5))
sns.barplot(x=list(test_data_dict.keys()), y=list(test_data_dict.values()))
plt.xlabel("Loại")
plt.ylabel("Số lượng hình ảnh")
plt.title("Số lượng hình ảnh của tập Test")
plt.show()

# Pie chart tổng thể
total_train = sum(train_data_dict.values())
total_test = sum(test_data_dict.values())
total_validate = sum(validate_data_dict.values())
total_files = total_train + total_test + total_validate

sizes = [(total_train / total_files) * 100, (total_test / total_files) * 100, (total_validate / total_files) * 100]
labels = ['Train', 'Test', 'Validate']
colors = ['orange', 'green', 'cyan']

plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.3f%%', startangle=90)
plt.title('Phân bổ dữ liệu của Dataset')
plt.show()

# Pie chart từng món
for food in ['BunBoHue', 'BanhChung', 'BanhXeo', 'BanhMi', 'BanhCuon']:
    train_count = train_data_dict[food]
    test_count = test_data_dict[food]
    validate_count = validate_data_dict[food]
    total_files = train_count + test_count + validate_count

    sizes = [(train_count / total_files) * 100, (test_count / total_files) * 100, (validate_count / total_files) * 100]
    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.3f%%', startangle=90)
    plt.title(f'Phân bổ dữ liệu của {food}')
    plt.show()

