import cdsapi
import xarray as xr
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from datetime import datetime,timedelta
import datetime as dt
import numpy as np
import pandas as pd
import pymannkendall as mk
import math
from numpy import cos as cos
from mpl_toolkits.basemap import Basemap
from scipy.stats import pearsonr

#%%
#Descargo los datos
c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'temperature',
        'pressure_level': '100',
        'year': [
            '1980', '1981', '1982',
            '1983', '1984', '1985',
            '1986', '1987', '1988',
            '1989', '1990', '1991',
            '1992', '1993', '1994',
            '1995', '1996', '1997',
            '1998', '1999', '2000',
            '2001', '2002', '2003',
            '2004', '2005', '2006',
            '2007', '2008', '2009',
            '2010', '2011', '2012',
            '2013', '2014', '2015',
            '2016', '2017', '2018',
            '2019', '2020',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'area': [
            -64, -57, -65,
            -56,
        ],
        'time': '00:00',
    },
    'download.nc')
#%% #Abro la data

data = xr.open_dataset("download.nc")

times = data.coords["time"].values

lon = data.coords["longitude"].values[-1] #longitud de marambio -56

lat = data.coords["latitude"].values[0] #latitud de marambio 64

data['t'].values.shape # (temperatura, latitud,longitud)

df = pd.DataFrame(data['t'].values[:,0,-1]-273 , index = data.coords["time"].values,columns=['t'])

df = df[:-12] #Descargue tmb el año 2020, lo elimino

#%% Parte 1 
#Figura 3
plt.plot(df['t'])
plt.ylim(-80,-40)
plt.xlabel('Año')
plt.ylabel('Temperatura (ºC)')

#%% Parte 2 

df['onda_anual'] = 0
df['sigma'] = 0

for m in range(1,13):
    df['onda_anual'][df.index.month==m] = df['t'][df.index.month==m].mean()
    df['sigma'][df.index.month==m]  = df['t'][df.index.month==m].std()
    
df['sinonda'] = df['t']-df['onda_anual']


# Figura 4
fig, ax = plt.subplots()
plt.plot(df['onda_anual'][0:13],label = 'Onda anual')
plt.ylabel('Temperatura (ºC)')
plt.ylim(-80,-35)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xlabel('Mes del año')
plt.xlim(df.index[0],df.index[11])
plt.legend(loc='best')

#Figura 5
fig, ax = plt.subplots()
plt.plot(df['sinonda']['2019'])
plt.ylabel('Anomalías de temperatura (ºC)')
plt.fill_between(df['2019'].index, df['sigma']['2019'],-df['sigma']['2019'],alpha=0.2 )
plt.axhline(0,color='k',linestyle='--')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xlabel('Mes del año 2019')
plt.xlim(df['2019'].index[0],df['2019'].index[-1])

#%% Parte 3, divido la serie en dos 

df1 = df[0:240]
df2 = df[240:]

e1m = df1['t'][df1.index.month==1].mean()
e1s = df1['t'][df1.index.month==1].std()
j1m = df1['t'][df1.index.month==7].mean()
j1s = df1['t'][df1.index.month==7].std()

e2m = df2['t'][df2.index.month==1].mean()
e2s= df2['t'][df2.index.month==1].std()
j2m = df2['t'][df2.index.month==7].mean()
j2s = df2['t'][df2.index.month==7].std()

#Test de medias usando T de student, gl= 18, alpha = 0.05, T_C = 1.7
t_est_enero = (e1m-e2m)/np.sqrt(((e1s**2)/20)+((e2s**2)/20))
t_est_julio = (j1m-j2m)/np.sqrt(((j1s**2)/20)+((j2s**2)/20))

print(t_est_enero)
print(t_est_julio)

#%% Parte 4, https://psl.noaa.gov/data/correlation/soi.data

data = np.loadtxt('soi.txt')
data = data.reshape(1,492)[0][:-12]

#Z_c = 1.96, test normal
corr, _ = pearsonr(data, np.array(df['t']))
Z = corr*np.sqrt((len(data)-2)/(1-corr**2))
print(Z)

df['soi'] = data
plt.plot(df.index, data)
plt.ylabel('SOI')
plt.xlabel('Tiempo')

#Z_C = 1.7 t-student, gl = 38, alpha = 0.05
corr_julio, _ = pearsonr(df['soi'][df.index.month==7],df['t'][df.index.month==7])
Z_julio = corr_julio*np.sqrt(38/(1-corr_julio**2))
print(Z_julio)

#Z_C = 1.7 t-student, gl = 38, alpha = 0.05
corr_enero, _ = pearsonr(df['soi'][df.index.month==1],df['t'][df.index.month==1])
Z_enero= corr*np.sqrt(38/(1-corr_enero**2))
print(Z_enero)


RC = np.sqrt((1.72**2)/(38+(1.72**2)))
print(RC)

#%% Parte 5, mapa de correlacion

julio = xr.open_dataset("correlation_julio.nc",decode_times=False)
enero = xr.open_dataset("correlation_enero.nc",decode_times=False)

lon = julio.coords['lon'].values
lat = julio.coords['lat'].values[1:-1]

for j in range(len(lon)):
    if lon[j] >180:
        lon[j] =lon[j]-360

lllon = -80
lllat = -75
urlon = -40
urlat = -55

#Figura 6, creo los mapas
fig = plt.figure(figsize=(10, 8))
ax0=fig.add_subplot(1, 2, 1)
ax0.set_title("Enero")
m1 = Basemap(projection = 'merc', llcrnrlon = lllon, llcrnrlat = lllat, urcrnrlon = urlon, urcrnrlat = urlat)
llons, llats = np.meshgrid(lon, lat)
xi, yi = m1(llons, llats)
datos1 = enero['air'].values
m1.drawcoastlines()
m1.drawmapboundary(color='k',linewidth=3.0,fill_color =None,zorder=None,ax=None)
m1.drawparallels(np.arange(lllat, urlat, 5.),color = 'black', linewidth = 0.5,labels=[1,0,0, 0])
m1.drawmeridians(np.arange(lllon, urlon, 10.),color = '0.25', linewidth = 0.5,labels=[0,0,0,1])
con = m1.pcolormesh(xi, yi, datos1[:,1:-1,:][0], cmap='seismic',vmin=-0.6, vmax=0.6)
plt.xlabel('Longitud',fontsize=10,labelpad=25)
plt.ylabel('Latitud',fontsize=10,labelpad=25)
lon1, lat1 = -58.38, -64.14 
xpt,ypt = m1(lon1,lat1)
m1.plot(xpt,ypt,'ko')
cbar = m1.colorbar(con,location='bottom',pad=0.8)
cbar.set_label("Coeficiente de correlación lineal" )

ax0 = fig.add_subplot(1, 2, 2)
ax0.set_title("Julio")
m = Basemap(projection = 'merc', llcrnrlon = lllon, llcrnrlat = lllat, urcrnrlon = urlon, urcrnrlat = urlat)
datos2 = julio['air'].values
m.drawcoastlines()
m.drawmapboundary(color='k',linewidth=3.0,fill_color =None,zorder=None,ax=None)
m.drawparallels(np.arange(lllat, urlat, 5.),color = 'black', linewidth = 0.5,labels=[1,0,0, 0])
m.drawmeridians(np.arange(lllon, urlon, 10.),color = '0.25', linewidth = 0.5,labels=[0,0,0,1])
con = m.pcolormesh(xi, yi, datos2[:,1:-1,:][0], cmap='seismic',vmin=-0.6, vmax=0.6)
plt.xlabel('Longitud',fontsize=10,labelpad=25)
m.plot(xpt,ypt,'ko')
cbar = m.colorbar(con,location='bottom',pad=0.8)
cbar.set_label("Coeficiente de correlación lineal" )
fig.tight_layout()

#%% Parte 6

prom_anual =[]
año = []
for y in range(1980,2020):
    prom= df['t'][df.index.year==y].mean()
    prom_anual.append(prom)
    año.append(y)

#Figura 7
plt.plot(año,prom_anual, label='Serie promedio anual')
plt.xlabel('Año')
plt.ylabel('Temperatura (ºC)')
plt.legend(loc='best')

test = mk.original_test(prom_anual)

N = len(df)
#tau calculado
tau_calculado = test.Tau 
print(tau_calculado)

# calculo el desvio de Tau 
desvio = np.sqrt((4*N+10)/(9*N*(N-1))) 

#tau tabla
tau_tabla = (1.96*desvio) 
print(tau_tabla)

if abs(tau_calculado)>tau_tabla:
    print('Rechazo H0')
else:
    print('No puedo rechazar H0')

#%% Parte 7, analisis armonico

valor = np.array(df['t'])

def analisis_armonico(valor):
    N = len(valor)
    K = int(N/2)
    PROM = np.mean(valor) 
    VAR = np.var(valor)
    VAR1 = VAR**0.5

    ## calculo los coeficientes de Fourier   

    NARM = np.zeros(K) #n de armonicos
    A = np.zeros(K) # A multiplica el seno
    B = np.zeros(K)  # B multiplica el conseno
    AM = np.zeros(K) # Amplitud
    C = np.zeros(K) # Varianza
    CA = np.zeros(K) # Varianza acumulda

    for i in range(0,K-1):
        NARM[i] = i+1 
        SUM = 0
        SAM = 0
        for j in range(0,N):
            SUM = SUM + valor[j]*np.sin((i+1)*2*np.pi*((j+1)/N))
            SAM = SAM + valor[j]*np.cos((i+1)*2*np.pi*((j+1)/N))      
        A[i] = 2*SUM/N
        B[i] = 2*SAM/N
        AM[i] = (A[i]**2+B[i]**2)**0.5
        C[i] = (((AM[i]**2)/2)/VAR)*100
        CA[i] = np.sum(C)
  
    SUM = 0	
    for j in range(0,N):
        SUM = SUM+(valor[j]*np.cos(K*2*np.pi*(j+1)/N))

    B[K-1] = SUM/N
    AM[K-1] = B[K-1]
    C[K-1] = ((AM[K-1]**2)/VAR)*100
    CA[K-1] =  CA[K-2]+C[K-1]
    A[K-1] = 0
    NARM[K-1] = K

    FIN = pd.DataFrame([A,B,AM, C, CA])
    FIN = FIN.T
    FIN.columns = ['A','B','AM','C','CA']  
    return FIN

FIN = analisis_armonico(valor)

#Figura 8
plt.plot(FIN['C'])
plt.xlabel('Nº de armónico')
plt.ylabel('Varianza explicada (%)')

###################### Constuyo y remuevo armonico que representa la onda anual

M = 40
PROM = np.mean(valor)   
##busco los coef A y B del armonico a filtrar

SUM = 0
SAM = 0

for J in range(0,N):
    SUM = SUM+valor[J]*np.sin(M*2*np.pi*(J+1)/N)
    SAM = SAM+valor[J]*np.cos(M*2*np.pi*(J+1)/N)

A = 2*SUM/N
B = 2*SAM/N

XS =  np.zeros(N)
FIL = np.zeros(N)

for J in range(0,N-1):
    XS[J] = PROM+A*np.sin(2*np.pi*M*((J+1)/N))+B*np.cos(2*np.pi*M*((J+1)/N))
    FIL[J] = valor[J]-XS[J]

plt.plot(df.index[0:-1],valor[0:-1], label='Serie original')
plt.plot(df.index[0:-1],XS[0:-1], label ="Armónico 41")
plt.xlabel('Tiempo')
plt.ylabel('Temperatura (ºC)')
plt.ylim(-80,-35)
plt.legend(loc='best')

plt.plot(df.index, FIL)
plt.ylabel('Anomalías de temperatura (ºC)')
plt.axhline(0,color='k',linestyle='--')
plt.xlabel('Tiempo')

################ análisis armonico a la serie filtrada

valor = np.array(FIL)
FIN1 = analisis_armonico(valor)

#Figura 9
plt.plot(FIN1['C'])
plt.xlabel('Nº de armónico')
plt.ylabel('Varianza explicada (%)')
#%% Parte 8

def analisis_espectral(ventana,confianza, df):
    pi=np.pi
    # Elija la Ventana: (1)Hann  (2)Hamming  (3)Parzen'
    ventana = ventana
    # Elija intervalos de confianza: (1)95% (2)90%
    intervalo = confianza
    n=int(df.size)
    q=1
    m=int(0.3*n)
    result=np.zeros([m+1,5])
    # Calcula las frecuencias y las guarda
    f=np.zeros([m+1])
    for k in range(m+1):
        f[k]=k/(2*m)
        result[k,0]=f[k]
    # Calcula intervalos de confianza
    gl=(2*n-(m/2))/m 
    if (intervalo==1):
        chiinf=(.06414329*gl**1.157371-.09347972)/(1.783669*gl**(-1.319044)+.1701124)
        chisup=(.9295456*gl**.5908365+3.231091)/(.8309965*gl**(-.3875086)-.006014)
    if (intervalo==2):
        chiinf=(.3559888*gl**2.408943-.3692226)/(.6930017*gl**1.299689+4.887728)
        chisup=(3.23178*gl**.9261549+19.64045)/(6.454432*gl**(-.8282872)+1.975337)
    #Calculo de la Ventana
    u=np.empty([m+1])
    if (ventana==1):
        for r in range(m):
            u[r]=.5*(1+cos(pi*r/m))

    if (ventana==2):
        for r in range(m):
            u[r]=.54+.46*cos(pi*r/m)

    if (ventana==3):
        for r in range(m):
            if (r<=(m/2)):
                u[r]=1-6*((r/m)**2)*(1-(r/m))
            if (r>(m/2)):
                u[r]=2*(1-(r/m))**3
    xm=df.data.mean()
    xd=np.empty([n]) 
    for i in range(n):
        xd[i]=df.data.iloc[i]-xm    
    l=int(.1*n)
    for i in range(1,l):
        c=0.5*(1-cos((i-1)*pi/(l-1)))
        xd[i-1]=c*xd[i-1]
        xd[n-i]=c*xd[n-i]

    Cov=np.zeros([m+1])
    for r in range(m):
        s=0
        for i in range(1,n-r):
            s=s + xd[i]*xd[i+r]
        Cov[r]=s/(n-r)
    R=Cov*u
    Gsum=0
    G=np.zeros([m+1])
    for k in range(m+1):
        s=0
        for r in range(1,m):
            s=s+R[r]*cos(pi*r*k/m)
        G[k]=2*(R[0]+2*s+R[m]*cos(pi*k)) #???
        Gsum=Gsum+G[k]
    Gmed=Gsum/(m+1)
    cor=np.corrcoef(df.data.iloc[1::].values,df.data.iloc[0:n-1].values)
    r1=cor[0,1]
    print('r1:', r1)
    if r1>0:
        R = (-1+1.96*np.sqrt(n-2))/(n-1)
        print('Rc:',R)
    else:
        R = (-1-1.96*np.sqrt(n-2))/(n-1)
        print('Rc:',R)
    if np.abs(r1)>np.abs(R):
        text = 'Ruido Rojo'
        print(text)
    else:
        text ='Ruido Blanco'
        print(text)
    Cn=np.zeros([m+1])
    Cnsup=np.zeros([m+1])
    Cninf=np.zeros([m+1])
    for k in range(m+1):
        Cn[k]=Gmed*((1-r1**2)/(1+r1**2-2*r1*cos(pi*k/m)))
        Cnsup[k]=Cn[k]*chisup/gl
        Cninf[k]=Cn[k]*chiinf/gl

    Cnn=np.zeros([m+1])
    Cnnsup=np.zeros([m+1])
    Cnninf=np.zeros([m+1])
    for k in range(m+1):
        Cnn[k]=Gmed
        Cnnsup[k]=Cnn[k]*chisup/gl
        Cnninf[k]=Cnn[k]*chiinf/gl

    result[:,1]=G
    if text =='Ruido Rojo':
        result[:,2]=Cninf
        result[:,3]=Cn
        result[:,4]=Cnsup
    else:
        result[:,2]=Cnninf
        result[:,3]=Cnn
        result[:,4]=Cnnsup
    result=pd.DataFrame(result,columns=['f','G','Cninf','Cn','Cnsup'])
    return result


df_aux =df.copy()
del df_aux['onda_anual']
del df_aux['sinonda']
del df_aux['soi']
del df_aux['sigma']
df_aux.rename(columns={"t": "data"},inplace = True)
                   

result = analisis_espectral(1,1, df_aux)
                
pos_max = result['G'].idxmax()
print('Frecuencia significativa: ', result['f'][pos_max], '1/mes')
print('Periodo significativo: ', 1/result['f'][pos_max],' meses')

#Figura 10
plt.plot(result['f'],result['G'],'k', label='Espectro empírico')
plt.plot(result['f'],result['Cn'],'-r',label='Espectro teórico')
plt.plot(result['f'],result['Cninf'],'--b', label='Bandas de significancia')
plt.plot(result['f'],result['Cnsup'],'--b')
plt.ylabel('G')
plt.xlim(0,0.25)
plt.xlabel('Frecuencia (mes$^{-1}$)')
plt.legend(loc='best')

resta = result['G']-result['Cnsup']>0
1/result['f'][resta]

#%% Parte 9, Wavelet

from __future__ import division
from matplotlib import pyplot
import pycwt as wavelet
from pycwt.helpers import find
import numpy

dt=1 
N=df_aux.data.size
t=np.arange(1,N+1,1)
mean=df_aux.data.mean()
std=df_aux.data.std()
var=std**2
norm_data=df_aux.data.values

mother = wavelet.Morlet(6)
s0 = 2 * dt  # Starting scale
dj = 1 / 12  # Twelve sub-octaves per octaves  ???
J = 7 / dj  # Seven powers of two with dj sub-octaves ???
alpha, _, _ = wavelet.ar1(df_aux.data.values)

wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(norm_data, dt, dj, s0, J,
                                                      mother)
iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std

power = (numpy.abs(wave)) ** 2
fft_power = numpy.abs(fft) ** 2
period = 1 / freqs
power /= scales[:, None]

signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                         significance_level=0.95,
                                         wavelet=mother)
sig95 = numpy.ones([1, N]) * signif[:, None]
sig95 = power / sig95

glbl_power = power.mean(axis=1)
dof = N - scales  # Correction for padding at edges
glbl_signif, tmp = wavelet.significance(var, dt, scales, 1, alpha,
                                        significance_level=0.95, dof=dof,
                                        wavelet=mother)

sel = find((period >= 2) & (period < 8))
Cdelta = mother.cdelta
scale_avg = (scales * numpy.ones((N, 1))).transpose()
scale_avg = power / scale_avg  # As in Torrence and Compo (1998) equation 24
scale_avg = var * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
scale_avg_signif, tmp = wavelet.significance(var, dt, scales, 2, alpha,
                                             significance_level=0.95,
                                             dof=[scales[sel[0]],
                                                  scales[sel[-1]]],
                                             wavelet=mother)
#Figura 12

pyplot.close('all')
pyplot.ioff()
figprops = dict(figsize=(11, 8), dpi=72)
fig = pyplot.figure(**figprops)


bx = pyplot.axes([0.1, 0.37, 0.65, 0.28])
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
con = bx.contourf(t, numpy.log2(period), numpy.log2(power), numpy.log2(levels),
            extend='both', cmap=pyplot.cm.viridis)
extent = [t.min(), t.max(), 0, max(period)]
bx.contour(t, numpy.log2(period), sig95, [-99, 8], colors='k', linewidths=2,
           extent=extent)
bx.fill(numpy.concatenate([t, t[-1:] + dt, t[-1:] + dt,
                           t[:1] - dt, t[:1] - dt]),
        numpy.concatenate([numpy.log2(coi), [1e-9], numpy.log2(period[-1:]),
                           numpy.log2(period[-1:]), [1e-9]]),
        'k', alpha=0.3, hatch='x')
bx.set_ylabel('Período (meses)')
Yticks = 2 ** numpy.arange(numpy.ceil(numpy.log2(period.min())),
                           numpy.ceil(numpy.log2(period.max())))
bx.set_yticks(numpy.log2(Yticks))
bx.set_yticklabels(Yticks)

bx.set_xticks([0,120,240,360,480])
bx.set_xticklabels([1980,1990,2000,2010,2020])
bx.set_xlim([0, 480])
bx.set_xlabel('Año')
bx.set_ylim(numpy.log2([period.min(), period.max()]))
cbar = fig.colorbar(con)
cbar.ax.set_ylabel('Espectro de poder de Wavelet')

