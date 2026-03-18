import h5py
import json

# Point this to your downloaded model
h5_path = 'models/Final_CNN_Transformer.h5'

print(f"Opening {h5_path} to fix Keras 3 compatibility issues...")

with h5py.File(h5_path, 'r+') as f:
    # 1. Read the internal model configuration
    model_config_string = f.attrs.get('model_config')
    
    if model_config_string is None:
        print("Error: No model_config found in the file.")
    else:
        model_config = json.loads(model_config_string)
        
        # 2. Loop through the layers and patch the InputLayer
        fixes_applied = 0
        for layer in model_config['config']['layers']:
            if layer['class_name'] == 'InputLayer':
                config = layer['config']
                
                # Change 'batch_shape' to 'batch_input_shape'
                if 'batch_shape' in config:
                    config['batch_input_shape'] = config.pop('batch_shape')
                    fixes_applied += 1
                    
                # Delete the 'optional' flag
                if 'optional' in config:
                    config.pop('optional')
                    fixes_applied += 1
        
        # 3. Save the patched configuration back into the h5 file
        if fixes_applied > 0:
            f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')
            print(f"✅ Successfully applied {fixes_applied} fixes to the config!")
        else:
            print("No fixes needed. The file looks perfectly compatible.")

print("Surgery complete. You can now run Streamlit.")