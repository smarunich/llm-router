# Create a custom router server

The router-server is a microservice responsible for taking the user prompt sent from the router controller and returning a useful prompt classification according to some policy, such as a task classification or an intent classification. The router controller uses that classification to pick one of the available LLMs as specified in the router controller config file.

In order to do this, a variety of prompt classification techniques can be employed. By default, blueprint includes classification policies based on task and complexity. These router-server models are served by NVIDIA Triton. The blueprint also includes an example of how to create your own policy. The example fine-tunes a model to classify prompts based on a user's intent assuming they are interacting with a support chatbot at a bank. The model is fine tuned using a [Banking dataset](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data).

To run the example:
- Ensure you have NVIDIA container toolkit installed along with the appropriate cuda drivers. To test compatability, you should be able to execute: `docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi`.
- Train a classification model by following the steps in `intent-router/intent_router.ipynb` for an example. In order for this notebook to run correctly, use `docker compose up router-builder` and then access the Jupyter Lab instance.
- The notebook will have you copy the trained model into `/model_repository` directory in the Jupyter Lab instance, which is mapped to the local `/routers` directory. 
- Update the router-controller configuration to include the new policy.
- Run `make up` to start the router-controller and router-server with your new model.


The fine-tuned model created in the example has this specification:


Base Model: DeBERTa-v3 
Source: Banking dataset from PolyAI-LDN
Categories: 77 distinct intents grouped into 10 main categories
- Billing and Payments
- Account Management
- Security and Fraud Prevention
- Transaction Support
- Technical Support
- Financial Planning
- International Services
- Customer Education
- Dispute Resolution
- Product Information