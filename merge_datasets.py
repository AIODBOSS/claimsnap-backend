import os
import zipfile
import yaml
import shutil
import glob

dataset_dir = os.path.abspath('dataset')
raw_zips_dir = os.path.abspath('raw_zips')

splits = ['train', 'valid']
for s in splits:
    os.makedirs(os.path.join(dataset_dir, s, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, s, 'labels'), exist_ok=True)

unified_classes = []
class_mapping = {}

zip_files = glob.glob(os.path.join(raw_zips_dir, '*.zip'))
total_images = 0
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

for zip_idx, zip_path in enumerate(zip_files):
    ds_name = f"ds_{zip_idx}"
    print(f"Processing dataset {zip_idx + 1}/{len(zip_files)}: {os.path.basename(zip_path)}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        namelist = zip_ref.namelist()
        
        # 1. Read data.yaml
        yaml_entry = next((n for n in namelist if n.replace('\\', '/').endswith('data.yaml')), None)
        if not yaml_entry:
            print(f"  Warning: No data.yaml found in {zip_path}, skipping.")
            continue
            
        data = yaml.safe_load(zip_ref.read(yaml_entry))
        class_mapping[ds_name] = {}
        names = data.get('names', [])
        
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys())]
            
        for old_id, class_name in enumerate(names):
            clean_name = str(class_name).strip().replace(' ', '_').lower()
            if clean_name not in unified_classes:
                unified_classes.append(clean_name)
            new_id = unified_classes.index(clean_name)
            class_mapping[ds_name][old_id] = new_id

        # Normalize all zip paths with forward slashes for fast lookup
        norm_map = {n.replace('\\', '/').lstrip('/'): n for n in namelist}

        # 2. Extract and match images/labels
        for orig_entry, real_zip_entry in norm_map.items():
            entry_lower = orig_entry.lower()
            
            if not entry_lower.endswith(image_extensions):
                continue
                
            target_split = None
            if 'train/images' in entry_lower:
                target_split = 'train'
            elif 'valid/images' in entry_lower or 'val/images' in entry_lower or 'test/images' in entry_lower:
                target_split = 'valid'
                
            if not target_split:
                continue

            total_images += 1
            ext = os.path.splitext(orig_entry)[1]
            safe_name = f"ds{zip_idx}_{target_split}_{total_images:06d}"
            
            dst_img = os.path.join(dataset_dir, target_split, 'images', safe_name + ext)
            dst_lbl = os.path.join(dataset_dir, target_split, 'labels', safe_name + '.txt')
            
            # Write image
            with open(dst_img, 'wb') as f_out:
                f_out.write(zip_ref.read(real_zip_entry))
                
            # Locate label counterpart
            lbl_candidate = orig_entry.replace('/images/', '/labels/')
            lbl_candidate = os.path.splitext(lbl_candidate)[0] + '.txt'
            
            if lbl_candidate in norm_map:
                lbl_content = zip_ref.read(norm_map[lbl_candidate]).decode('utf-8', errors='ignore')
                with open(dst_lbl, 'w', encoding='utf-8') as f_out:
                    for line in lbl_content.splitlines():
                        parts = line.strip().split()
                        if not parts:
                            continue
                        try:
                            old_id = int(parts[0])
                            new_id = class_mapping[ds_name].get(old_id, old_id)
                            f_out.write(f"{new_id} {' '.join(parts[1:])}\n")
                        except ValueError:
                            continue
            else:
                open(dst_lbl, 'w').close()

# 3. Output master data.yaml
master_yaml = {
    'path': dataset_dir.replace('\\', '/'),
    'train': 'train/images',
    'val': 'valid/images',
    'nc': len(unified_classes),
    'names': unified_classes
}

with open(os.path.join(dataset_dir, 'data.yaml'), 'w', encoding='utf-8') as f:
    yaml.dump(master_yaml, f, sort_keys=False)

print("\n=== Dataset Merge Complete ===")
print(f"Total Processed Images: {total_images}")
print(f"Total Unique Classes: {len(unified_classes)}")
print(f"Classes: {', '.join(unified_classes)}\n")
