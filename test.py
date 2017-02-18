
x=0
def generator(iterable, batch_size=512):
    """
    @data: pd.DataFrame
    """
    x += 1
    yield x


for batch in generator(1):
	print(batch)

for batch in generator(1):
	print(batch)