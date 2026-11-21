import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet201
import numpy as np
import tensorflow as tf
import cv2
from sklearn.metrics import roc_curve, auc

# -----------------------------模型設定--------------------------------
# 載入keras模型(更換輸出圖片尺寸)
model = DenseNet201(include_top=False,
                 weights='imagenet',
                 input_tensor=Input(shape=(224, 224, 3))
                 )

# 定義輸出層
x = model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=model.input, outputs=predictions)

# 編譯模型
model.compile(optimizer=Adam(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# ---------------------------設置callbacks----------------------------
# 設定earlystop條件
estop = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

# 設定模型儲存條件
checkpoint = ModelCheckpoint('Densenet201_checkpoint_ntuh.h5', verbose=1,
                          monitor='val_loss', save_best_only=True,
                          mode='min')

# 設定lr降低條件(0.001 → 0.0002 → 0.00004 → 0.00001)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                           patience=5, mode='min', verbose=1,
                           min_lr=1e-6)
# -----------------------------設置資料集--------------------------------
# 設定ImageDataGenerator參數(路徑、批量、圖片尺寸)
train_dir = r'E:\Eric\traintriplet_NewNTUHYO-densenet\trdensnet'
valid_dir = r'E:\Eric\traintriplet_NewNTUHYO-densenet\valdensnet'
test_dir = r'E:\Eric\traintriplet_NewNTUHYO-densenet\tsdensnet-ntuh497'
batch_size = 64
target_size = (224, 224)

# 設定批量生成器
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   rotation_range=35,
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   shear_range=0.3, 
                                   zoom_range=0.5,
                                   fill_mode="constant") #constant nearest

val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

# 讀取資料集+批量生成器，產生每epoch訓練樣本
train_generator = train_datagen.flow_from_directory(train_dir,
                                      target_size=target_size,
                                      batch_size=batch_size)

valid_generator = val_datagen.flow_from_directory(valid_dir,
                                      target_size=target_size,
                                      batch_size=batch_size)

test_generator = test_datagen.flow_from_directory(test_dir,
                                      target_size=target_size,
                                      batch_size=batch_size,
                                      shuffle=False)

# -----------------------------開始訓練模型------------------------------
# 重新訓練權重
history = model.fit_generator(train_generator,
                   epochs=100, verbose=1,
                   steps_per_epoch=train_generator.samples//batch_size,
                   validation_data=valid_generator,
                   validation_steps=valid_generator.samples//batch_size,
                   callbacks=[checkpoint, estop, reduce_lr])
# -----------------------儲存模型、紀錄學習曲線------------------------
# 儲存模型
model.save('Densenet201_ntuh.h5')

# 畫出acc學習曲線
acc = history.history['accuracy']
epochs = range(1, len(acc) + 1)
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend(loc='lower right')
plt.grid()
# 儲存acc學習曲線
plt.savefig('acc.png')
# plt.show()

# 畫出loss學習曲線
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc='upper right')
plt.grid()
# 儲存loss學習曲線
plt.savefig('loss.png')
# plt.show()
#-------------臨床測試結果-----------------------------------
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=batch_size,
    shuffle=False
)

# 取得test類別
class_names = list(test_generator.class_indices.keys())

# 評估模型測試資料集表現
test_loss, test_acc = model.evaluate_generator(test_generator)
print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')

# 預測結果
test_preds = model.predict_generator(test_generator)


# 計算precision、recall
from sklearn.metrics import classification_report

print('Classification Report:')
report = classification_report(test_generator.classes,
                               test_preds.argmax(axis=1),
                               target_names=class_names)
print(report)
#=====儲存結果==========================================
filenames = test_generator.filenames
predictions = {}
for filename, pred in zip(filenames, test_preds):
    predicted_class_index = np.argmax(pred)
    predicted_class = class_names[predicted_class_index]
    probabilities = {class_name: prob for class_name, prob in zip(class_names, pred)}
    predictions[filename] = (predicted_class, probabilities)

# 檔名、預測類別，各類比重
with open('test_preds.txt', 'w') as f:
    for filename, (predicted_class, probabilities) in predictions.items():
        prob_str = ", ".join([f"{class_name}: {prob:.6f}" for class_name, prob in probabilities.items()])
        # print(f"File: {filename}, Predicted class: {predicted_class}")
        # print(f"Probabilities: {prob_str}")
        # print()  # 添加一個空行以提高可讀性
        f.write(filename+","+predicted_class+","+prob_str+"\n")
#==============================================
print("訓練完畢")
#============畫冷熱圖==================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path=r"E:\Eric\traintriplet_NewNTUHYO-densenet", alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")

    cv2.imwrite(cam_path, superimposed_img)

test_generator.reset()

num_samples = test_generator.samples

class_names = list(test_generator.class_indices.keys())

# 遍历整个测试集
for i in range(num_samples):
    # 获取单个图像和标签
    img, label = test_generator.next()
    img = img[0]  # 因为next()返回的是一个批次，我们只需要第一个（也是唯一一个）图像
    
    # 获取原始文件名并移除类别前缀
    full_filename = test_generator.filenames[i]
    class_name, original_filename = os.path.split(full_filename)
    
    # 构建新的文件名
    base_name = os.path.splitext(original_filename)[0]
    gradcam_filename = f"{base_name}_gradcam.jpg"
    
    img_array = np.expand_dims(img, axis=0)
    
    # 生成热力图 conv5_block32_2_conv
    heatmap = make_gradcam_heatmap(img_array, model, 'conv5_block32_concat')
    
    # 保存并显示Grad-CAM
    save_and_display_gradcam(os.path.join(test_dir, full_filename), 
                             heatmap, 
                             os.path.join('HeatMapIMG', class_name, gradcam_filename))

    # print(f"Grad-CAM for image {original_filename} saved as {gradcam_filename}")

    # 可选：每处理100张图片打印一次进度
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1} / {num_samples} images")  
#============ROC======================================================
unique_classes = np.unique(test_generator.classes)
# 將 y_test 轉換為 one-hot 編碼
y_test = np.zeros((len(test_generator.classes), len(unique_classes)))
for i, label in enumerate(test_generator.classes):
    y_test[i, label] = 1
print("Unique classes:", unique_classes)
print("test_generator.classes:", test_generator.classes)
print("New y_test shape:", y_test.shape)
print("test_preds shape:", test_preds.shape)

# 計算每個類別的ROC曲線和AUC
fpr = dict()
tpr = dict()
roc_auc = dict()

n_classes = len(unique_classes)

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], test_preds[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 繪製所有ROC曲線
plt.figure(figsize=(10, 8))

colors = ['blue', 'red', 'green', 'yellow', 'purple']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
             label=f'ROC curve of class {unique_classes[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('ROC_curve.png')
# plt.show()

# 輸出每個類別的AUC分數
for i, class_name in enumerate(unique_classes):
    print(f'AUC for class {class_name}: {roc_auc[i]:.2f}')
#=======================================================================================
print("ROC輸出完畢")