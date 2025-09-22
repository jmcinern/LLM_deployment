# LLM_deployment
To deploy the Qomhrá bilingual Irish-English LLM. Considerations: Memory, Chat Templating, Hosting.

## Quantize 
- Activation-aware weight quantization
    - Identifies salient weights with calibtating on forward pass
    - Uses a scaling factor to quantize more important weights less and less imortant weights more.