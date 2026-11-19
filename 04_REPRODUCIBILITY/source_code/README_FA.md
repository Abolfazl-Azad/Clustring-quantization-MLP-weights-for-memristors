# پروژه نرم‌افزاری VGG8 با PyTorch

این پروژه یک نسخه مشخص از VGG8 برای دسته‌بندی ۱۰ کلاس CIFAR-10 را پیاده‌سازی می‌کند. معماری شامل ۵ لایه Conv و ۳ لایه Fully Connected است و بعد از هر Conv از Batch Normalization و ReLU استفاده می‌کند.

## معماری

- ورودی: `[N, 3, 32, 32]`
- Conv1: `3 -> 64`، سپس ReLU و MaxPool
- Conv2: `64 -> 192`، سپس ReLU و MaxPool
- Conv3: `192 -> 384`، سپس ReLU
- Conv4: `384 -> 256`، سپس ReLU
- Conv5: `256 -> 256`، سپس ReLU و MaxPool
- Flatten: `256*4*4 = 4096`
- FC6: `4096 -> 256`، ReLU
- FC7: `256 -> 128`، ReLU
- FC8: `128 -> 10` logits

توجه: نام VGG8 در همه پروژه‌ها به یک معماری واحد اشاره نمی‌کند؛ این README مشخصات همین پیاده‌سازی را تعریف می‌کند.

## ۱. نصب روی Windows

Python 3.10 یا جدیدتر پیشنهاد می‌شود. در PowerShell، داخل پوشه پروژه:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

اگر فعال‌سازی محیط مجازی به‌خاطر Execution Policy مسدود شد، برای همان پنجره PowerShell اجرا کن:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

اگر GPU انویدیا و CUDA سازگار داری، فرمان نصب مخصوص سیستم خودت را از صفحه رسمی PyTorch انتخاب کن: https://pytorch.org/get-started/locally/

## ۲. تست سریع مدل

```powershell
python model.py
python smoke_test.py
```

این تست شکل تانسورها و یک گام Forward/Backward را کنترل می‌کند؛ هنوز دیتاست دانلود یا آموزش کامل اجرا نمی‌شود.

## ۳. آموزش

```powershell
python train.py --epochs 20 --batch-size 128
```

در اجرای اول، CIFAR-10 دانلود می‌شود؛ بنابراین دسترسی اینترنت لازم است، مگر اینکه دیتاست از قبل در پوشه `data` موجود باشد.

پارامترهای مفید:

```powershell
python train.py --epochs 5 --batch-size 64 --device cpu
python train.py --epochs 30 --batch-size 128 --num-workers 2
python train.py --epochs 20 --lr 0.001 --output-dir outputs/my_experiment
```

روی Windows اگر DataLoader با worker مشکل داشت، `--num-workers 0` را نگه دار. برای GPU، `--device auto` به‌صورت خودکار CUDA را انتخاب می‌کند، اگر PyTorch نسخه سازگار CUDA را تشخیص دهد.

خروجی‌های پیش‌فرض:

- `outputs/vgg8_cifar10/best.pt`: بهترین مدل بر اساس دقت اعتبارسنجی
- `outputs/vgg8_cifar10/last.pt`: checkpoint آخرین Epoch
- `outputs/vgg8_cifar10/history.csv`: Loss و Accuracy هر Epoch

## ۴. ارزیابی روی Test Set

بعد از پایان آموزش:

```powershell
python evaluate.py --checkpoint outputs/vgg8_cifar10/best.pt
```

این مرحله دقت کلی و دقت هر کلاس را گزارش می‌کند.

## ۵. پیش‌بینی روی یک تصویر

```powershell
python predict.py --image "C:\path\to\image.jpg" --checkpoint outputs/vgg8_cifar10/best.pt --top-k 3
```

مدل برای تصاویر کوچک CIFAR-10 آموزش می‌بیند. برای عکس‌های واقعی و بزرگ، صرفاً Resize کردن به 32×32 می‌تواند اطلاعات زیادی را از بین ببرد؛ بنابراین پیش‌بینی روی عکس‌های معمولی الزاماً قابل اعتماد نیست.

## نکات مهم

- خروجی مدل **logits** است. هنگام آموزش، آن را مستقیماً به `CrossEntropyLoss` بده؛ قبلش Softmax اعمال نکن.
- برای استنتاج، مدل در `eval()` قرار می‌گیرد و `torch.no_grad()` استفاده می‌شود.
- این کد VGG8 را آموزش می‌دهد؛ وزن‌های آموزش‌دیده داخل پروژه قرار نگرفته‌اند و پس از اجرای `train.py` تولید می‌شوند.
- برای تبدیل به FPGA، پس از تثبیت دقت نرم‌افزار باید مرحله جداگانه‌ای برای Quantization، صادرکردن وزن‌ها و ساخت golden reference انجام شود.
