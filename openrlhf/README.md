## Running Commands

### ImplicitPRM
To train Implicit PRM, you can run following commands
```
cd ./tasks
bash run_ce.sh
bash run_dpo.sh
```
The above scripts will automatically download the dataset  `Windy0822/ultrainteract_math_rollout` from huggingface and transform it to the format of OpenRLHF pipeline, which will be saved at the path indicated by the `--dataset` argument.

Other argument settings are similar to the OpenRLHF package. 

