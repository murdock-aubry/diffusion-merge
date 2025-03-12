import torch 

model1_name = "sd1.4-cocotuned"
model2_name = "sd1.4-dogtuned"

model1 = torch.load(f'/w/383/murdock/hidden_reps/{model1_name}/representations.pt')
model2 = torch.load(f'/w/383/murdock/hidden_reps/{model2_name}/representations.pt')

prompts1 = model1["prompts"]
inputs1 = model1["inputs"]
outputs1 = model1["outputs"]

prompts2 = model2["prompts"]
inputs2 = model2["inputs"]
outputs2 = model2["outputs"]

npoints1 = len(prompts1)
npoints2 = len(prompts2)

if npoints1 != npoints2:
    print("number of points are not equal")
else:
    npoints = npoints1

if prompts1 != prompts2:
    print("prompt misalignment")
else:
    prompts = prompts1


nlayers = inputs1[0].shape[0]

# colapse inputs and outputs
for iprompt in range(npoints):
    inputs1[iprompt] = torch.flatten(inputs1[iprompt], start_dim=-2)
    inputs2[iprompt] = torch.flatten(inputs2[iprompt], start_dim=-2)

    outputs1[iprompt] = torch.flatten(outputs1[iprompt], start_dim=-2)
    outputs1[iprompt] = torch.flatten(outputs1[iprompt], start_dim=-2)
