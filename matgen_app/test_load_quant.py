import os, sys
sys.path.insert(0, 'matgen_app')
os.environ['MATGEN_DEVICE'] = 'cpu'

output = open('matgen_app/load_quant_out.txt', 'w')
output.write('Loading with 4-bit quantization...\n')
output.flush()

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # 4-bit quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype='float16',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )

    output.write('Loading tokenizer...\n')
    output.flush()
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
    output.write('Tokenizer loaded\n')
    output.flush()

    output.write('Loading model with 4-bit quantization...\n')
    output.flush()
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-7B-Instruct',
        quantization_config=quantization_config,
        device_map='auto',
        trust_remote_code=True,
    )
    output.write(f'Model loaded! Params: {sum(p.numel() for p in model.parameters())/1e9:.1f}B\n')
    output.flush()

    # Save for later use
    output.write('Saving model...\n')
    output.flush()
    model.save_pretrained('matgen_app/models/qwen_vasp_4bit')
    tokenizer.save_pretrained('matgen_app/models/qwen_vasp_4bit')
    output.write('Model saved to matgen_app/models/qwen_vasp_4bit\n')
    output.flush()

except Exception as e:
    output.write(f'Error: {e}\n')
    import traceback
    traceback.print_exc(file=output)
    output.flush()
output.close()
print("Done - check load_quant_out.txt")