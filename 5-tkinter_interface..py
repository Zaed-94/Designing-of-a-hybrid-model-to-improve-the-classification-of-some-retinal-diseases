import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, Canvas, Scrollbar, Frame
from PIL import Image, ImageTk
import datetime
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import uuid
import shutil
import pandas as pd
import cv2

# تغيير الدليل الحالي إلى موقع السكريبت لضمان قراءة الملفات من نفس المجلد الذي يوجد به هذا الملف
import os
import sys

if getattr(sys, 'frozen', False):
    # التطبيق يعمل كملف EXE
    base_path = os.path.dirname(sys.executable)
else:
    # التطبيق يعمل كسكريبت Python عادي
    base_path = os.path.dirname(os.path.abspath(__file__))

os.chdir(base_path)
print("الدليل الحالي:", os.getcwd())

# ------------------------------
# دوال Grad-CAM المساعدة
# ------------------------------
def get_last_conv_layer(model):
    """
    تعيد اسم آخر طبقة تلافيفية (Conv2D) في الموديل.
    تُستخدم لاحقاً لحساب Grad-CAM.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    تنشئ خريطة Grad-CAM باستخدام الصورة المُعدة (img_array)
    والموديل واسم آخر طبقة تلافيفية (last_conv_layer_name).
    - pred_index: فهرس التصنيف المطلوب (إذا لم يُحدد، يتم اختيار التصنيف الأعلى).
    تُرجع مصفوفة الخريطة الحرارية.
    """
    # إنشاء موديل جزئي يُخرج مخرجات آخر طبقة تلافيفية والناتج النهائي للموديل
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        # حساب مخرجات الطبقة الأخيرة والتنبؤات
        conv_outputs, predictions = grad_model(img_array)
        # تحديد الفهرس إذا لم يُحدد من قبل
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        # حساب الخسارة بالنسبة للتصنيف المحدد
        loss = predictions[:, pred_index]
    # حساب التدرجات بالنسبة لمخرجات الطبقة التلافيفية
    grads = tape.gradient(loss, conv_outputs)
    # تجميع التدرجات (حساب المتوسط عبر الأبعاد المكانة)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    # حساب الخريطة الحرارية عن طريق ضرب مخرجات الطبقة بالتدرجات المجمعة
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    # تفعيل الخريطة الحرارية بقيمة صفرية للقيّم السالبة وتطبيعها ليكون المدى من 0 إلى 1
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def apply_top_percent_threshold(heatmap, top_percent=10):
    """
    تُبقي على أعلى top_percent% من القيم في الخريطة الحرارية
    وتضبط الباقي إلى صفر.
    تساعد على إبراز المناطق الأكثر دلالة.
    """
    cutoff = np.percentile(heatmap, 100 - top_percent)
    heatmap[heatmap < cutoff] = 0.0
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap

def overlay_heatmap(img, heatmap, alpha=0.3):
    """
    تُدمج الخريطة الحرارية (heatmap) مع الصورة الأصلية (img) 
    باستخدام شفافية (alpha).
    يُفترض أن img عبارة عن مصفوفة NumPy (بالصيغة BGR كما يستخدمها OpenCV).
    """
    # تغيير حجم الخريطة لتطابق أبعاد الصورة
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # تحويل الخريطة الحرارية إلى قيم صحيحة تتراوح من 0 إلى 255
    heatmap = np.uint8(255 * heatmap)
    # تطبيق خريطة ألوان (COLORMAP_JET) على الخريطة الحرارية
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # دمج الصورة الأصلية والخريطة باستخدام وزن كل منهما
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed_img

# ------------------------------
# إعدادات ملفات Excel ومجلد التخزين
# ------------------------------
EXCEL_FILE_SINGLE = 'classification_results_single.xlsx'  # ملف نتائج التصنيف الفردي
EXCEL_FILE_FOLDER = 'classification_results_folder.xlsx'  # ملف نتائج التصنيف للمجلد
OUTPUT_FOLDER = 'classified_images'  # مجلد حفظ نسخ الصور المصنفة

# ------------------------------
# تحميل النماذج والملفات المطلوبة
# ------------------------------
try:
    model = tf.keras.models.load_model('best_model_g.keras')
    print("تم تحميل نموذج CNN بنجاح.")
except Exception as e:
    print("خطأ في تحميل نموذج CNN:", e)

try:
    # إنشاء مستخرج ميزات من الطبقة قبل الأخيرة في النموذج
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    print("تم إنشاء مستخرج الميزات بنجاح.")
except Exception as e:
    print("خطأ في إنشاء مستخرج الميزات:", e)

try:
    svm_model = joblib.load('Final_model.pkl')
    print("تم تحميل نموذج SVM بنجاح.")
except Exception as e:
    print("خطأ في تحميل نموذج SVM:", e)

try:
    selected_indices = np.load('selected_indices.npy')
    print("تم تحميل selected_indices بنجاح:", selected_indices)
except Exception as e:
    print("خطأ في تحميل selected_indices:", e)

# ------------------------------
# إعداد حجم الصورة والفئات
# ------------------------------
IMAGE_SIZE = (128, 128)  # الحجم المتوقع للصور عند الإدخال للنموذج
class_names = ['CRVO', 'DME', 'OTHER']  # قائمة أسماء الفئات المتوقعة

# ------------------------------
# دوال تحديث ملفات Excel
# ------------------------------
def update_excel_single(data_dict):
    """
    تحدث ملف Excel الخاص بالتصنيف الفردي بإضافة سجل جديد من البيانات.
    إذا لم يكن الملف موجودًا يتم إنشاؤه.
    """
    try:
        if os.path.exists(EXCEL_FILE_SINGLE):
            df = pd.read_excel(EXCEL_FILE_SINGLE, engine="openpyxl")
        else:
            df = pd.DataFrame(columns=['ID', 'File', 'Class', 'Probability (%)', 'Date', 'Time', 'PatientName', 'Age', 'Eye'])
        df = df.append(data_dict, ignore_index=True)
        df.to_excel(EXCEL_FILE_SINGLE, index=False, engine="openpyxl")
        print("تم تحديث ملف Excel الخاص بالتصنيف الفردي.")
    except Exception as e:
        print("خطأ في تحديث ملف Excel الخاص بالتصنيف الفردي:", e)

def update_excel_folder(data_list):
    """
    تحدث ملف Excel الخاص بتصنيف المجلد بإضافة سجلات جديدة من البيانات.
    إذا لم يكن الملف موجودًا يتم إنشاؤه.
    """
    try:
        df_new = pd.DataFrame(data_list)
        if os.path.exists(EXCEL_FILE_FOLDER):
            df_existing = pd.read_excel(EXCEL_FILE_FOLDER, engine="openpyxl")
            df = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df = df_new
        df.to_excel(EXCEL_FILE_FOLDER, index=False, engine="openpyxl")
        print("تم تحديث ملف Excel الخاص بتصنيف المجلد.")
    except Exception as e:
        print("خطأ في تحديث ملف Excel الخاص بتصنيف المجلد:", e)

# ------------------------------
# دالة تجهيز الصورة: تغيير الحجم وتطبيعها وإضافة بعد الدُفعة
# ------------------------------
def preprocess_image(image_path, target_size=IMAGE_SIZE):
    """
    تقرأ الصورة من المسار المحدد، تغير حجمها لتتناسب مع النموذج،
    تُحوّلها إلى مصفوفة باستخدام img_to_array وتطبع قيم البكسل بإجراء التطبيع،
    وتُضيف بعد الدُفعة (batch dimension) لتصبح جاهزة للإدخال في النموذج.
    """
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img
    except Exception as e:
        print("خطأ في تجهيز الصورة:", e)
        return None, None

# ------------------------------
# دالة استخراج الميزات باستخدام مستخرج الميزات المُنشأ
# ------------------------------
def extract_selected_features(image_path):
    """
    تُجهّز الصورة باستخدام preprocess_image ثم تمرّرها عبر feature_extractor.
    بعد ذلك تختار الميزات المطلوبة باستخدام selected_indices وتُعيدها مع نسخة PIL للصورة.
    """
    processed, pil_img = preprocess_image(image_path)
    if processed is None:
        return None, None
    try:
        features = feature_extractor.predict(processed)
        print(f"شكل الميزات للصورة {os.path.basename(image_path)}: {features.shape}")
        selected_features = features[:, selected_indices]
        return selected_features, pil_img
    except Exception as e:
        print("خطأ أثناء استخراج الميزات:", e)
        return None, None

# ------------------------------
# دالة حفظ نسخة من الصورة في مجلد النتائج
# ------------------------------
import random

def save_image_to_folder(image_path, predicted_class):
    """
    تنسخ الصورة من المسار المحدد إلى مجلد فرعي داخل OUTPUT_FOLDER يحمل اسم التصنيف.
    يُنشئ اسم الملف من اسم التصنيف متبوعًا برقم عشوائي مكوّن من 4 أرقام والامتداد الأصلي.
    """
    # إنشاء المجلد الفرعي الخاص بالتصنيف داخل OUTPUT_FOLDER
    class_folder = os.path.join(OUTPUT_FOLDER, predicted_class)
    os.makedirs(class_folder, exist_ok=True)
    
    ext = os.path.splitext(image_path)[1]
    # توليد رقم عشوائي من 0 إلى 9999 وصياغته بأربع خانات (مثلاً 0057)
    unique_number = random.randint(0, 9999)
    unique_filename = f"{predicted_class}_{unique_number:04d}{ext}"
    target_path = os.path.join(class_folder, unique_filename)
    
    try:
        shutil.copy(image_path, target_path)
        print(f"تم حفظ الصورة في: {target_path}")
        return unique_filename
    except Exception as e:
        print("خطأ في حفظ الصورة:", e)
        return None


# ------------------------------
# دالة تصنيف صورة واحدة وعرض النتائج مع Grad-CAM
# ------------------------------
def classify_single_image():
    """
    تفتح مربع حوار لاختيار صورة من الجهاز.
    تقوم بتجهيز الصورة واستخراج الميزات ثم تصنيفها باستخدام نموذج SVM.
    بعدها تُحسب خريطة Grad-CAM باستخدام نموذج CNN وتُعرض الصورة الأصلية وصورة Grad-CAM جنبًا إلى جنب.
    كما يتم عرض التصنيف والنسبة وإمكانية إدخال بيانات المريض مع حفظ النتيجة.
    """
    file_path = filedialog.askopenfilename(title="اختر صورة", filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
    if not file_path:
        return

    # تجهيز الصورة وحساب الميزات
    processed, pil_img = preprocess_image(file_path)
    selected_feats, _ = extract_selected_features(file_path)
    if selected_feats is None:
        messagebox.showerror("خطأ", "تعذر استخراج الميزات من الصورة.")
        return

    # التصنيف باستخدام نموذج SVM
    try:
        if hasattr(svm_model, "predict_proba"):
            proba = svm_model.predict_proba(selected_feats)[0]
            predicted_idx = np.argmax(proba)
            probability = proba[predicted_idx]
        else:
            predicted_idx = svm_model.predict(selected_feats)[0]
            probability = None
    except Exception as e:
        messagebox.showerror("خطأ", f"تعذر التصنيف: {e}")
        return

    predicted_class = class_names[predicted_idx] if predicted_idx < len(class_names) else str(predicted_idx)
    print(f"التصنيف: {predicted_class}، الاحتمالية: {probability*100 if probability is not None else 'غير معروف'}%")
    
    # حساب Grad-CAM باستخدام آخر طبقة تلافيفية في النموذج
    last_conv_layer_name = get_last_conv_layer(model)
    if last_conv_layer_name is None:
        messagebox.showerror("خطأ", "لم يتم العثور على طبقة تلافيفية في النموذج.")
        return
    heatmap = make_gradcam_heatmap(processed, model, last_conv_layer_name)
    heatmap = apply_top_percent_threshold(heatmap, top_percent=10)
    
    # تحويل الصورة الأصلية إلى صيغة cv2 (BGR) وحساب overlay مع Grad-CAM
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gradcam_cv = overlay_heatmap(img_cv, heatmap, alpha=0.3)
    # إعادة تحويل الصورة الناتجة إلى RGB ثم إلى صورة PIL
    gradcam_rgb = cv2.cvtColor(gradcam_cv, cv2.COLOR_BGR2RGB)
    gradcam_img = Image.fromarray(gradcam_rgb)
    
    # إنشاء نافذة لعرض النتائج
    result_window = tk.Toplevel(root)
    result_window.title("نتيجة التصنيف للصورة الواحدة")
    result_window.configure(bg="#f7f7f7")
    
    # عرض التاريخ والوقت الحالي
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tk.Label(result_window, text=f"التاريخ والوقت: {now}", font=("Helvetica", 10),
             bg="#f7f7f7", fg="#333333").pack(pady=5)
    
    # إنشاء إطار رئيسي لتقسيم العرض إلى عمودين: صورة أصلية وصورة Grad-CAM
    main_frame = tk.Frame(result_window, bg="#f7f7f7")
    main_frame.pack(padx=10, pady=10, fill="both", expand=True)
    
    # العمود الأيسر: عرض الصورة الأصلية
    left_frame = tk.Frame(main_frame, bg="#f7f7f7")
    left_frame.grid(row=0, column=0, padx=10, pady=10)
    pil_img_thumb = pil_img.copy()
    pil_img_thumb.thumbnail((250,250))
    orig_tk = ImageTk.PhotoImage(pil_img_thumb)
    tk.Label(left_frame, image=orig_tk, bg="#f7f7f7").pack()
    left_frame.img_tk = orig_tk  # الاحتفاظ بالمرجع
    
    # العمود الأيمن: عرض صورة Grad-CAM
    right_frame = tk.Frame(main_frame, bg="#f7f7f7")
    right_frame.grid(row=0, column=1, padx=10, pady=10)
    gradcam_thumb = gradcam_img.copy()
    gradcam_thumb.thumbnail((250,250))
    gradcam_tk = ImageTk.PhotoImage(gradcam_thumb)
    tk.Label(right_frame, image=gradcam_tk, bg="#f7f7f7").pack()
    right_frame.img_tk = gradcam_tk  # الاحتفاظ بالمرجع
    
    # عرض نص التصنيف والنسبة أسفل الصور
    result_text = f"التصنيف: {predicted_class}\n"
    if probability is not None:
        result_text += f"نسبة الاحتمالية: {probability*100:.2f}%"
    tk.Label(result_window, text=result_text, font=("Helvetica", 14), bg="#f7f7f7", fg="#0066CC").pack(pady=5)
    
    # إطار لإدخال بيانات المريض وحفظ النتيجة (اختياري)
    entry_frame = tk.Frame(result_window, bg="#f7f7f7")
    entry_frame.pack(pady=10, padx=10)
    tk.Label(entry_frame, text="اسم المريض:", bg="#f7f7f7", fg="#333333").grid(row=0, column=0, sticky="e", padx=5, pady=2)
    patient_name_entry = tk.Entry(entry_frame, font=("Helvetica", 10))
    patient_name_entry.grid(row=0, column=1, padx=5, pady=2)
    tk.Label(entry_frame, text="العمر:", bg="#f7f7f7", fg="#333333").grid(row=1, column=0, sticky="e", padx=5, pady=2)
    age_entry = tk.Entry(entry_frame, font=("Helvetica", 10))
    age_entry.grid(row=1, column=1, padx=5, pady=2)
    tk.Label(entry_frame, text="العين (يمين/يسار):", bg="#f7f7f7", fg="#333333").grid(row=2, column=0, sticky="e", padx=5, pady=2)
    eye_entry = tk.Entry(entry_frame, font=("Helvetica", 10))
    eye_entry.grid(row=2, column=1, padx=5, pady=2)
    
    # دالة لحفظ النتيجة في ملف Excel واستكمال العملية
    def save_result():
        image_id = str(uuid.uuid4())
        saved_filename = save_image_to_folder(file_path, predicted_class)
        if saved_filename is None:
            messagebox.showerror("خطأ", "تعذر حفظ الصورة.")
            return
        now_date = datetime.datetime.now().strftime("%Y-%m-%d")
        now_time = datetime.datetime.now().strftime("%H:%M:%S")
        data = {
            'ID': image_id,
            'File': saved_filename,
            'Class': predicted_class,
            'Probability (%)': probability*100 if probability is not None else None,
            'Date': now_date,
            'Time': now_time,
            'PatientName': patient_name_entry.get(),
            'Age': age_entry.get(),
            'Eye': eye_entry.get()
        }
        update_excel_single(data)
        messagebox.showinfo("تم", f"تم حفظ النتيجة:\n{result_text}\nبيانات المريض: {data['PatientName']}, {data['Age']}, {data['Eye']}")
        result_window.destroy()
    
    tk.Button(result_window, text="حفظ النتائج", command=save_result,
              bg="#0066CC", fg="white", activebackground="#004C99", font=("Helvetica", 11, "bold")
             ).pack(pady=10)

# ------------------------------
# دالة تصنيف مجلد الصور وعرض النتائج مع Grad-CAM لكل صورة
# ------------------------------
def classify_folder():
    """
    تفتح مربع حوار لاختيار مجلد الصور، ثم تستخدم os.walk لاستعراض كافة الصور داخل المجلد والمجلدات الفرعية.
    لكل صورة يتم تجهيزها، استخراج الميزات، التصنيف وحساب Grad-CAM.
    تُجمّع النتائج لتحديث ملف Excel ويُعرض كل سجل في شبكة مع عرض الصورة الأصلية والـ Grad-CAM.
    """
    folder_path = filedialog.askdirectory(title="اختر مجلد الصور")
    if not folder_path:
        return
    # البحث في المجلد وجميع المجلدات الفرعية عن ملفات الصور بالامتدادات المحددة
    image_files = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif')):
                image_files.append(os.path.join(dirpath, f))
                
    print("عدد الصور في المجلد:", len(image_files))
    if not image_files:
        messagebox.showwarning("تنبيه", "لا توجد صور في المجلد المحدد.")
        return

    results = []         # لتجميع البيانات لتحديث ملف Excel
    images_display = []  # لتجميع بيانات العرض (الصورة الأصلية والـ Grad-CAM)
    
    # الحصول على اسم آخر طبقة تلافيفية مرة واحدة من النموذج
    last_conv_layer_name = get_last_conv_layer(model)
    if last_conv_layer_name is None:
        messagebox.showerror("خطأ", "لم يتم العثور على طبقة تلافيفية في النموذج.")
        return
    
    # تكرار لكل ملف صورة في القائمة
    for img_path in image_files:
        processed, pil_img = preprocess_image(img_path)
        if processed is None:
            print(f"تعذر تجهيز الصورة: {img_path}")
            continue
        selected_feats, _ = extract_selected_features(img_path)
        if selected_feats is None:
            print(f"تعذر استخراج الميزات للصورة: {img_path}")
            continue
        try:
            if hasattr(svm_model, "predict_proba"):
                proba = svm_model.predict_proba(selected_feats)[0]
                predicted_idx = np.argmax(proba)
                probability = proba[predicted_idx]
            else:
                predicted_idx = svm_model.predict(selected_feats)[0]
                probability = None
        except Exception as e:
            print(f"خطأ في تصنيف الصورة {img_path}: {e}")
            continue
        predicted_class = class_names[predicted_idx] if predicted_idx < len(class_names) else str(predicted_idx)
        image_id = str(uuid.uuid4())
        saved_filename = save_image_to_folder(img_path, predicted_class)
        now_date = datetime.datetime.now().strftime("%Y-%m-%d")
        now_time = datetime.datetime.now().strftime("%H:%M:%S")
        data = {
            'ID': image_id,
            'File': saved_filename,
            'Class': predicted_class,
            'Probability (%)': probability*100 if probability is not None else None,
            'Date': now_date,
            'Time': now_time,
            'PatientName': "",
            'Age': "",
            'Eye': ""
        }
        results.append(data)
        
        # حساب Grad-CAM للصورة
        heatmap = make_gradcam_heatmap(processed, model, last_conv_layer_name)
        heatmap = apply_top_percent_threshold(heatmap, top_percent=10)
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gradcam_cv = overlay_heatmap(img_cv, heatmap, alpha=0.3)
        gradcam_rgb = cv2.cvtColor(gradcam_cv, cv2.COLOR_BGR2RGB)
        gradcam_img = Image.fromarray(gradcam_rgb)
        images_display.append((pil_img, gradcam_img, predicted_class, probability))
    
    print("عدد النتائج المجمعة:", len(results))
    if results:
        update_excel_folder(results)
        messagebox.showinfo("تم", f"تم حفظ النتائج في {EXCEL_FILE_FOLDER}")
    else:
        messagebox.showwarning("تنبيه", "لم يتم تصنيف أي صورة بنجاح.")
    
    display_images_grid(images_display)

# ------------------------------
# دالة عرض نتائج التصنيف للمجلد بنمط شبكي
# لكل سجل تُعرض الصورة الأصلية وصورة Grad-CAM مع نص التصنيف والاحتمالية
# ------------------------------
def display_images_grid(images):
    result_window = tk.Toplevel(root)
    result_window.title("نتائج تصنيف المجلد")
    result_window.configure(bg="#f7f7f7")
    
    canvas = Canvas(result_window, bg="#f7f7f7")
    scrollbar = Scrollbar(result_window, orient="vertical", command=canvas.yview)
    frame_grid = Frame(canvas, bg="#f7f7f7")
    
    # قائمة لتخزين المراجع للصور بحيث لا تُحذف من الذاكرة
    frame_grid.img_refs = []
    
    frame_grid.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    
    canvas.create_window((0, 0), window=frame_grid, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    cols = 2  # في كل صف خليتان: صورة أصلية و Grad-CAM
    row = 0
    col = 0
    
    for orig_img, gradcam_img, pred_class, prob in images:
        cell_frame = tk.Frame(frame_grid, bg="#f7f7f7", bd=1, relief="solid", padx=5, pady=5)
        # تجهيز الصورة الأصلية للعرض
        orig_thumb = orig_img.copy()
        orig_thumb.thumbnail((150,150))
        orig_tk = ImageTk.PhotoImage(orig_thumb)
        tk.Label(cell_frame, image=orig_tk, bg="#f7f7f7").pack()
        # تجهيز Grad-CAM للعرض
        gradcam_thumb = gradcam_img.copy()
        gradcam_thumb.thumbnail((150,150))
        gradcam_tk = ImageTk.PhotoImage(gradcam_thumb)
        tk.Label(cell_frame, image=gradcam_tk, bg="#f7f7f7").pack(pady=5)
        # عرض نص التصنيف والاحتمالية
        text = (f"التصنيف: {pred_class}\nالاحتمالية: {prob*100:.2f}%" if prob is not None 
                else f"التصنيف: {pred_class}")
        tk.Label(cell_frame, text=text, font=("Helvetica", 12), bg="#f7f7f7", fg="#0066CC", wraplength=150, justify="center").pack(pady=5)
        cell_frame.grid(row=row, column=col, padx=10, pady=10)
        # الاحتفاظ بالمراجع حتى لا تُجمع القمامة
        cell_frame.orig_tk = orig_tk
        cell_frame.gradcam_tk = gradcam_tk
        
        col += 1
        if col >= cols:
            col = 0
            row += 1
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    result_window.geometry("600x800")

# ------------------------------
# دالة تحديث الوقت والتاريخ في الواجهة الرئيسية
# ------------------------------
def update_datetime():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    datetime_label.config(text=now)
    root.after(1000, update_datetime)

# ------------------------------
# إعداد الواجهة الرئيسية باستخدام Tkinter
# ------------------------------
root = tk.Tk()
root.title("واجهة تصنيف الصور")
root.geometry("350x300")
root.configure(bg="#e6f2ff")

# عرض الوقت في الركن العلوي الأيمن
datetime_label = tk.Label(root, text="", font=("Helvetica", 10), bg="#e6f2ff", fg="#003366")
datetime_label.place(relx=1.0, rely=0.0, anchor="ne", x=-10, y=10)
update_datetime()

# عنوان الواجهة
tk.Label(root, text="(OCT) تصنيف الصور", font=("Helvetica", 16, "bold"),
         bg="#e6f2ff", fg="#003366").pack(pady=30)

# زر لتصنيف صورة واحدة مع عرض Grad-CAM
tk.Button(root, text="تصنيف صورة واحدة", width=25, command=classify_single_image,
          bg="#0066CC", fg="white", activebackground="#004C99", font=("Helvetica", 12, "bold")
         ).pack(pady=10)

# زر لتصنيف جميع الصور داخل مجلد مع عرض Grad-CAM لكل صورة
tk.Button(root, text="تصنيف مجلد الصور", width=25, command=classify_folder,
          bg="#0066CC", fg="white", activebackground="#004C99", font=("Helvetica", 12, "bold")
         ).pack(pady=10)

# بدء الحلقة الرئيسية للواجهة
root.mainloop()
