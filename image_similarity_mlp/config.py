color_channel = 3  
image_size = 32    
hidden_size = 256  
output_size = 64   
temperature = 0.07           
num_epochs = 100
log_interval = 10
embedding_dim = 10

# Mapping from number label to descriptive text
label_map = {
    0: "Un avión",
    1: "Un automóvil",
    2: "Un pájaro",
    3: "Un gato",
    4: "Un ciervo",
    5: "Un perro",
    6: "Una rana",
    7: "Un caballo",
    8: "Un barco",
    9: "Un camión"
}
#Join all dict in one array. Split them in words (tokenize). Take only unique words and calculate its length
vocab_size = len(set(" ".join(label_map.values()).split()))

