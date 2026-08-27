#making a multi attention head 
class multiattenttionmodel(nn.Module):
  #this class will be used to have multiple attention heads run in parrallel
  def __init__(self , numberofhead , head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size)for _ in range(numberofhead)]) 
    #creates multiple head instances 
    #those heads are then stored into the modulelist as seperate submodules allowign for their parameters of each head to be tracked
    #each head then has its own set of results such that each head has its own key , query and value array each beign the numerical form of communication between tokens 
    #in this example the multiple attention head makes it so that 4 seperate heads are ran simultaenously
    
  
  def forward(self,x):
    return torch.cat([h(x) for h in self.heads] , dim = -1)


class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocabsize,numberofembed) #this has each idex vlaue has accossiated values in the embdded table
    self.position_embedding_table = nn.Embedding(block_size , numberofembed)
    self.languagemodelhead = nn.Linear(numberofembed , vocabsize)
    self.sa_heads = multiattenttionmodel(4 , numberofembed//4) #4 seperate headers , each header has the T = 8
    #as we are making 4 headers and we need each head size to be 8 (headsize is the T value representing number of tokens in input)
    #and number fo embed is the total number of values in the input x

  def forward(self, inputx , target = None ):
    B , T = inputx.shape
    token_embed = self.token_embedding_table(inputx)    #(batch , time , count)
    position_embed = self.position_embedding_table(torch.arange(T , device = device))
    x = token_embed + position_embed #with numbers accosiated to both position and identity each value now is unique and so locating and identifying is done so muhc easier
    x = self.sa_heads(x)
    logits = self.languagemodelhead(x) #(batch , time , vocabsize) #transforms the data into the correct size to then be learnt from

    #gets the tokens , accosiated embeded table values and runs them through a linear layer
    if target == None :
      loss = None
    else:
      batchsize , tokenlength,channel = logits.shape
      logits = logits.view(batchsize*tokenlength , channel)
      #this logic also applies to the target values
      target = target.view(batchsize*tokenlength)
      loss = F.cross_entropy(logits , target) #calculates a value from the logits and targets , this is loss function whcih rates how right or wrong the guesses were

    return logits , loss

  def generate(self,idx,max_newtokens):
    #this method will be used to actually use the loss and logits by having the model generate the next values from the first value for the amount of new token wanted to be generated in that row
    for new_data in range(max_newtokens):
      idx_trim = idx[: , -block_size:]
      logits , loss = self(idx_trim)
      probabilities = F.softmax(logits , dim = -1)
      next_token = torch.multinomial(probabilities , num_samples = 1) #gets token with the highest probability
      idx = torch.cat((idx , next_token) , dim = 1) #adds that tokne with the highest probability onto x


    return idx


Bigram = BigramLanguageModel().to(device)
optimiser = torch.optim.Adam(Bigram.parameters() , lr = lr )

#now we will see if this optimiser actually work
batchsize = 32
for steps in range(25001):
  xb , yb = get_batch("train")
  logits , loss = Bigram.forward(xb , yb)
  optimiser.zero_grad() #emtpies out previous results of backward propogation so that it does not influence backward propogation
  loss.backward() #perform backward propogation to then get the amount you wantt to change the paramterrs
  optimiser.step() #performs that step to change the parameters
  if steps % 1000 == 0:
    print(f"step: {steps} loss: {loss.item()}")
