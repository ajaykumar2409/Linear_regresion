

```python
# simplest example of showing the relation between the area and price.
import numpy as np
import matplotlib.pyplot as plt
a = np.arange(1,11)
b = [300,350,500,700,800,850,900,900,1000,1200]
plt.plot(a,b,'ro')
plt.ion()
```


![png](output_0_0.png)



```python
import numpy as np
import matplotlib.pyplot as plt
def estimate_coff(a,b):
    n = np.size(a) # size of the dataset
    mean_x,mean_y = np.mean(a),np.mean(b)
    # calculate the error
    error_xy = np.sum(b*a-n*mean_y*mean_x)
    error_xx = np.sum(a*a-n*mean_x*mean_x)
    b_1 = error_xy/error_xx
    b_0 = mean_y -b_1*mean_x
    return (b_0,b_1)

def ploting_the_line(a,b,c):
    plt.scatter(a,b,color='k')
    y_pred = b[0]+b[1]*a
    plt.plot(a,y_pred,color='green')
    plt.xlabel('Size')
    plt.ylabel('Cost')
    plt.show()
    
def main():
    a = np.arange(1,11)
    b = [300,350,500,700,800,850,900,900,1000,1200]
    c = estimate_coff(a,b)
    print("Estimated cofficent:\n{}\n{}".format(b[0],b[1]))
    ploting_the_line(a,b,c)
    plt.show()
```


```python
if __name__ == "__main__":
    main()
```

    Estimated cofficent:
    300
    350



![png](output_2_1.png)

