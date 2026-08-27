#starting with the dataset to be trained on
!pip install datasets -q
from datasets import load_dataset
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ds = load_dataset("roneneldan/TinyStories")
#print(ds)
#print(ds['train'][0])
text = " ".join([item['text'] for item in ds['train'].select(range(5000))])
print(len(text))  # number of characters
print(text[:1000])

#getting all unique characters that occur in this text
characters =  sorted(list(set(text))) #sorting all uniques charcaters in the text
vocabsize = len(characters)
print("".join(characters))
print(vocabsize)

# Now we need to tokenise characters
# tokenising characters converts characters into integers, which can then be processed
# This can be done by mapping characters to integers#
stoi = {character:index for index , character in enumerate(characters)}
itos = {index:character for index , character in enumerate(characters)}
encode = lambda string: [stoi[character] for character in string] #iterates through the passed in string and then passes it into stoi to encode it
decode = lambda letter: "".join([itos[integer] for integer in letter]) #iterates through list of values representing each character and runs them through
#the decoder
print(encode("Hi my name is Aliff")) #outputs the values for all the strings
print(decode(encode("Hi my name is Aliff"))) #outputs for charcaters for all the numbers


#now to encode the entire text dataset and turn it into a PyTorch tensor
#this will allow the model to handle tensors
import torch
data = torch.tensor(encode(text) , dtype = torch.long).to(device)
print(data.shape)
print(data[:1000])

#now to encode the entire text dataset and turn it into a PyTorch tensor
#this will allow the model to handle tensors
import torch
data = torch.tensor(encode(text) , dtype = torch.long).to(device)
print(data.shape)
print(data[:1000])

#as inputting large pieces of text into a neural network is computationally inefficient and expensive, we can split the training se
#into smaller pieces, that way we can train the model on different pieces of data whilst being reasonable
fraction_of_data = 8
traindata[:fraction_of_data+1] #plus one has been done due to our model predicting the next letters using the previous letters
#specifically this is doen by honing into the data before, for example to predict 61 the mdoel focuses on 36 and to predict 52 the model uses 36 and 61
#and so there are 8 pieces of data to be focused on to predict 1

# This can be seen with this
x = traindata[:fraction_of_data]
y = traindata[1:fraction_of_data + 1]

for integer in range(fraction_of_data):
  print(integer)
  print(f"what is being focuses on {x[:integer+1]}")
  print(f"what the prediction is {y[integer]}")


# Object-oriented version of creation of self-attention head
block_size = 8
numberofembed = 32
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module):
  #implements one head of self-attention
  def __init__(self , head_size):
    super().__init__()
    self.key = nn.Linear(C , head_size , bias = False)
    self.query = nn.Linear(C , head_size, bias = False)
    self.value = nn.Linear(C , head_size , bias = False)
    self.register_buffer("tril",torch.tril(torch.ones(block_size , block_size))) #creates the triangle
    #but as the tril is not a parameter tril is just a buffer: a "temp" variable.
    # This is only needed due to us needing to hide future data

  def forward(self,x):
    B , T , C = x.shape #4 batches , 8 timecounts and 32 channels
    #4 batches
    # Each batch has 8 tokens
    # Each token has 32 numbers associated with it
    k = self.key(x)
    q = self.query(x)
    weight = q @ k.transpose(-2,-1) * C**-0.5  #produces a B , T ,T dimension array
    #*C**-0.5 is scaled attention
    # Scaled attention is done to keep values returned from the dot product within a reasonable range
    #this is needed as weights can get very large and we don't want certain tokens dominating in terms of communication, drowning out the other tokens
    weight = weight.masked_fill(self.tril[:T , :T] == 0 ,float("-inf"))
    weight = F.softmax(weight, dim = -1)
    v = self.value(x)
    output = weight @ v
    return output

class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocabsize,numberofembed) #this has each index value has associated values in the embedded table
    self.position_embedding_table = nn.Embedding(block_size , numberofembed)
    self.languagemodelhead = nn.Linear(numberofembed , vocabsize)
    self.sa_head = Head(numberofembed)

  def forward(self, inputx , target = None ):
    B , T = inputx.shape
    token_embed = self.token_embedding_table(inputx)    #(batch , time , count)
    position_embed = self.position_embedding_table(torch.arange(T , device = device))
    x = token_embed + position_embed #with numbers associated to both position and identity each value now is unique and so locating and identifying is done so muhc easier
    x = self.sa_head(x)
    logits = self.languagemodelhead(x) #(batch , time , vocabsize) #transforms the data into the correct size to then be learnt from

    #gets the tokens , accosiated embeded table values and runs them through a linear layer
    if target == None :
      loss = None
    else:
      batchsize , tokenlength,channel = logits.shape
      logits = logits.view(batchsize*tokenlength , channel)
      #this logic also applies to the target values
      target = target.view(batchsize*tokenlength)
      loss = F.cross_entropy(logits , target) #calculates a value from the logits and targets, this is the loss function which rates how right or wrong the guesses were

    return logits , loss

  def generate(self,idx,max_newtokens):
    # This method will be used to actually use the loss and logits by having the model generate the next values from the first value for the amount of new tokens wanted to be generated in that row
    for new_data in range(max_newtokens):
      idx_trim = idx[: , -block_size:]
      logits , loss = self(idx_trim)
      probabilities = F.softmax(logits , dim = -1)
      next_token = torch.multinomial(probabilities , num_samples = 1) #gets token with the highest probability
      idx = torch.cat((idx , next_token) , dim = 1) #adds that tokne with the highest probability onto x


    return idx

Bigram = BigramLanguageModel().to(device)
optimiser = torch.optim.Adam(Bigram.parameters() , lr = lr )

# Now we will see if this optimiser actually work
batchsize = 32
for steps in range(25001):
  xb , yb = get_batch("train")
  logits , loss = Bigram.forward(xb , yb)
  optimiser.zero_grad() # empties out previous results of backward propagation so that it does not influence backward propagation
  loss.backward() #perform backward propagation to then get the amount you want to change the parameters
  optimiser.step() #performs that step to change the parameters
  if steps % 1000 == 0:
    print(f"step: {steps} loss: {loss.item()}")







































