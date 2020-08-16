# Trainer

## Examples

Running diagnostics for `mxnet` and `hardware`

```bash
 python ./train.py --diagnose mxnet

```

```bash
 python ./train.py --diagnose hardware
```

## cleanup
```
ps aux | grep python | awk '{print $2}' | xargs kill
```

## Training

Usage

```bash
python ./train.py --train detector --data-dir ../data/flickr/trainingset
```

### Training / Test / Validation folder structure.

Training set directory structure requires three folders `train_data` `test_data` `val_data`.
Each folder should contain a subfolder that is the class the image belogs to with images present inside that folder

```bash
folder
    -- test_data
        -- class_a
            -- img_0001.png
            -- img_0002.png
        -- class_b
            -- img_0001.png
            -- img_0002.png
    -- train_data
        -- class_a
            -- img_0001.png
            -- img_0002.png
        -- class_b
            -- img_0001.png
            -- img_0002.png
    -- val_data
        -- class_a
            -- img_0001.png
            -- img_0002.png
        -- class_b
            -- img_0001.png
            -- img_0002.png
```

## glueon-cv version patch

Patched version of `glueon-cv`

Small patch here to allow using of version 2.0.0 of mxnet as I am building mxnet localy and not using binary-package.

```text
 Legacy mxnet==2.0.0 detected, some modules may not work properly. mxnet>=1.4.0,<2.0.0 is required. You can use pip to upgrade mxnet `pip install -U 'mxnet>=1.4.0,<2.0.0'` or `pip install -U 'mxnet-cu100>=1.4.0,<2.0.0'` 
```


```bash
diff --git a/gluoncv/utils/version.py b/gluoncv/utils/version.py
index de77051..73be6ee 100644
--- a/gluoncv/utils/version.py
+++ b/gluoncv/utils/version.py
@@ -32,7 +32,7 @@ def _require_mxnet_version(mx_version, max_mx_version='2.0.0'):
         import mxnet as mx
         from distutils.version import LooseVersion
         if LooseVersion(mx.__version__) < LooseVersion(mx_version) or \
-            LooseVersion(mx.__version__) >= LooseVersion(max_mx_version):
+            LooseVersion(mx.__version__) > LooseVersion(max_mx_version):
             version_str = '>={},<{}'.format(mx_version, max_mx_version)
             msg = (
                 "Legacy mxnet=={0} detected, some modules may not work properly. "

```
