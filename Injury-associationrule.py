import pandas as pd
import matplotlib.pyplot as plt

#   Setting the path for file export
path='C:\\Users\\leech\\OneDrive\\Desktop\\'

#   Data Import
data = pd.read_csv('C:/Users/leech/Downloads/severeinjury 010115-310118.csv',encoding = "ISO-8859-1")
eventData = pd.read_csv('C:/Users/leech/OneDrive/Desktop/Injury/event.csv',encoding = "ISO-8859-1")
natureData = pd.read_csv('C:/Users/leech/OneDrive/Desktop/Injury/nature.csv',encoding = "ISO-8859-1")
partData = pd.read_csv('C:/Users/leech/OneDrive/Desktop/Injury/part.csv',encoding = "ISO-8859-1")
sourceData = pd.read_csv('C:/Users/leech/OneDrive/Desktop/Injury/source.csv',encoding = "ISO-8859-1")

data.dtypes

#   Drop irrelevant columns
data.drop(["ID","Employer","Address1","Address2","City","State","Zip","Latitude","Longitude",
           "Inspection","Final Narrative","Secondary Source"],axis=1,inplace=True)

#   Preparation of column for alteration
data[["Nature","Part of Body","Event","Source"]]=data[["Nature","Part of Body","Event","Source"]].astype(str)

############################################   Feature Engineering Start ############################################

#   Extract the month and day of week from the date
data['EventDate'] = pd.to_datetime(data['EventDate'])
data['Month']=data['EventDate'].dt.strftime('%b')
data['DayOfWeek']=data['EventDate'].dt.day_name()
print(data['DayOfWeek'].values.ravel())
data.drop(["EventDate"],axis=1,inplace=True)

#   Altering the Hospitalized column
data['Hospitalized']=data['Hospitalized'].apply(lambda x: "Hospitalized" if x>0 else "Not-Hospitalized")

#   Altering the Amputation column
data['Amputation']=data['Amputation'].apply(lambda x: "Amputated" if x>0 else "Not-Amputated")
data['Amputation'].values.ravel()

#   Altering the NAICS column
data['NAIC-edited']=0
data.loc[data["Primary NAICS"].str[0:2]=="11","NAIC-edited"]="Agriculture-Forestry-Fishing-and-Hunting"
data.loc[data["Primary NAICS"].str[0:2]=="21","NAIC-edited"]="Mining"
data.loc[data["Primary NAICS"].str[0:2]=="22","NAIC-edited"]="Utilities"
data.loc[data["Primary NAICS"].str[0:2]=="23","NAIC-edited"]="Construction"
data.loc[(data["Primary NAICS"].str[0:2]=="31")|(data["Primary NAICS"].str[0:2]=="32")|(data["Primary NAICS"].str[0:2]=="33"),"NAIC-edited"]="Manufacturing"
data.loc[data["Primary NAICS"].str[0:2]=="42","NAIC-edited"]="Wholesale-Trade"
data.loc[(data["Primary NAICS"].str[0:2]=="44")|(data["Primary NAICS"].str[0:2]=="45"),"NAIC-edited"]="Retail-Trade"
data.loc[(data["Primary NAICS"].str[0:2]=="48")|(data["Primary NAICS"].str[0:2]=="49"),"NAIC-edited"]="Transportation-and-Warehousing"
data.loc[data["Primary NAICS"].str[0:2]=="51","NAIC-edited"]="Information"
data.loc[data["Primary NAICS"].str[0:2]=="52","NAIC-edited"]="Finance-and-Insurance"
data.loc[data["Primary NAICS"].str[0:2]=="53","NAIC-edited"]="Real-Estate-Rental-and-Leasing"
data.loc[data["Primary NAICS"].str[0:2]=="54","NAIC-edited"]="Professional-Scientific-and-Technical-Services"
data.loc[data["Primary NAICS"].str[0:2]=="55","NAIC-edited"]="Management-of-Companies-and-Enterprises"
data.loc[data["Primary NAICS"].str[0:2]=="56","NAIC-edited"]="Administrative-and-Support-and-Waste-Management-and-Remediation-Services"
data.loc[data["Primary NAICS"].str[0:2]=="61","NAIC-edited"]="Educational-Services"
data.loc[data["Primary NAICS"].str[0:2]=="62","NAIC-edited"]="Health-Care-and-Social-Assistance"
data.loc[data["Primary NAICS"].str[0:2]=="71","NAIC-edited"]="Arts-Entertainment-and-Recreation"
data.loc[data["Primary NAICS"].str[0:2]=="72","NAIC-edited"]="Accommodation-and-Food-Services"
data.loc[data["Primary NAICS"].str[0:2]=="81","NAIC-edited"]="Other-Services-(except-Public-Administration)"
data.loc[data["Primary NAICS"].str[0:2]=="92","NAIC-edited"]="Public-Administration"


#   Altering the Nature column
data['Nature-edited']=0
data.loc[data["Nature"].str[0]=="1","Nature-edited"]="Traumatic-Injuries-and-Disorders"
data.loc[data["Nature"].str[0]=="2","Nature-edited"]="Systemic-Diseases-and-Disorders"
data.loc[data["Nature"].str[0]=="3","Nature-edited"]="Infectious-and-Parasitic-Diseases"
data.loc[data["Nature"].str[0]=="4","Nature-edited"]="Neoplasms-Tumors-and-Cancers"
data.loc[data["Nature"].str[0]=="5","Nature-edited"]="Symptoms-Signs-and-Ill-defined-Conditions"
data.loc[data["Nature"].str[0]=="6","Nature-edited"]="Other-Diseases-Conditions-and-Disorders"
data.loc[data["Nature"].str[0]=="7","Nature-edited"]="Exposures-to-Disease—No-Illness-Incurred"
data.loc[data["Nature"].str[0]=="8","Nature-edited"]="Multiple-Diseases-Conditions-and-Disorders"
data.loc[data["Nature"].str[0]=="9","Nature-edited"]="Nonclassifiable"


#   Altering the event column
data['Event-edited']=0
data.loc[data["Event"].str[0]=="1","Event-edited"]="Violence-and-Other-Injuries-by-Persons-or-Animals"
data.loc[data["Event"].str[0]=="2","Event-edited"]="Transportation-Incidents"
data.loc[data["Event"].str[0]=="3","Event-edited"]="Fires-and-Explosions"
data.loc[data["Event"].str[0]=="4","Event-edited"]="Falls-Slips-Trips"
data.loc[data["Event"].str[0]=="5","Event-edited"]="Exposure-to-Harmful-Substances-or-Environments"
data.loc[data["Event"].str[0]=="6","Event-edited"]="Contact-with-Objects-and-Equipment"
data.loc[data["Event"].str[0]=="7","Event-edited"]="Overexertion-and-Bodily-Reaction"
data.loc[data["Event"].str[0]=="9","Event-edited"]="Nonclassifiable"

#   Altering the source column
data['Source-edited']=0
data.loc[data["Source"].str[0]=="1","Source-edited"]="Chemicals-and-Chemical-Products"
data.loc[data["Source"].str[0]=="2","Source-edited"]="Containers-Furniture-and-Fixtures"
data.loc[data["Source"].str[0]=="3","Source-edited"]="Machinery"
data.loc[data["Source"].str[0]=="4","Source-edited"]="Parts-and-Materials"
data.loc[data["Source"].str[0]=="5","Source-edited"]="Persons-Plants-Animals-and-Minerals"
data.loc[data["Source"].str[0]=="6","Source-edited"]="Structures-and-Surfaces"
data.loc[data["Source"].str[0]=="7","Source-edited"]="Tools-Instruments-and-Equipment"
data.loc[data["Source"].str[0]=="8","Source-edited"]="Vehicles"
data.loc[(data["Source"].str[0]=="9")&(data["Source"]!="9999"),"Source-edited"]="Other-Sources"
data.loc[data["Source"]=="9999","Source-edited"]="Nonclassifiable"

#   Altering the body part column
data['Body-edited']=0
data.loc[data["Part of Body"].str[0]=="1","Body-edited"]="Head"
data.loc[data["Part of Body"].str[0]=="2","Body-edited"]="Neck"
data.loc[data["Part of Body"].str[0]=="3","Body-edited"]="Trunk"
data.loc[data["Part of Body"].str[0]=="4","Body-edited"]="Upper-Extremities"
data.loc[data["Part of Body"].str[0]=="5","Body-edited"]="Lower-Extremities"
data.loc[data["Part of Body"].str[0]=="6","Body-edited"]="Body-Systems"
data.loc[data["Part of Body"].str[0]=="8","Body-edited"]="Multiple-Body-Parts"
data.loc[(data["Part of Body"].str[0]=="9")&(data["Part of Body"]!="9999"),"Body-edited"]="Other-Body-Parts"
data.loc[data["Part of Body"]=="9999","Body-edited"]="Nonclassifiable"

#########################   Feature Engineering to include hierarchy 2 items   #########################

eventData=eventData.loc[eventData["Hierarchy_level"]==2]
natureData=natureData.loc[natureData["Hierarchy_level"]==2]
partData=partData.loc[partData["Hierarchy_level"]==2]
sourceData=sourceData.loc[sourceData["Hierarchy_level"]==2]

mapping1 = {i: eventData.loc[eventData['CASE_CODE']==i,'CASE_CODE_TITLE'].iloc[0] for i in eventData['CASE_CODE']}
mapping1[99]="Nonclassifiable"

mapping2 = {i: natureData.loc[natureData['CASE_CODE']==i,'CASE_CODE_TITLE'].iloc[0] for i in natureData['CASE_CODE']}
mapping2[99]="Nonclassifiable"
mapping2[7]="Exposure to Disease-No illness incurred"
mapping2[8]="Multiple diseases, conditions and disorders"

mapping3 = {i: partData.loc[partData['CASE_CODE']==i,'CASE_CODE_TITLE'].iloc[0] for i in partData['CASE_CODE']}
mapping3[99]="Nonclassifiable"
mapping3[6]="BodySystem"

mapping4 = {i: sourceData.loc[sourceData['CASE_CODE']==i,'CASE_CODE_TITLE'].iloc[0] for i in sourceData['CASE_CODE']}
mapping4[99]="Nonclassifiable"

data['Event']=data['Event'].str[0:2].astype(int)
data['Nature']=data['Nature'].str[0:2].astype(int)
data['Source']=data['Source'].str[0:2].astype(int)
data['Part of Body']=data['Part of Body'].str[0:2].astype(int)

data['Event']=data['Event'].map(mapping1)
data['Nature']=data['Nature'].map(mapping2)
data['Source']=data['Source'].map(mapping4)
data['Part of Body']=data['Part of Body'].map(mapping3)

# #########################   Feature Engineering to include hierarchy 2 items ends  #########################

#   function to export dataset to sas
def exportingCSV(dataset, fileName):
    #   Converting to SAS format
    dataset = dataset.set_index('UPA').stack()
    dataset = dataset.reset_index()
    dataset = dataset[['UPA',0]]
    dataset = dataset.rename(columns={'UPA':'Index',0:'Target'})

    #   Data Export
    dataset.to_csv(path+fileName+".csv", index=False)


#   Looking at the cluster which is hospitalized and amputated
firstCluster = data.loc[(data['Hospitalized']=='Hospitalized')&(data['Amputation']=='Amputated')]

#   Manufacturing got the most
data.dtypes
naics = firstCluster.groupby('NAIC-edited')['UPA'].nunique()
naics
naics.sum()
ax=naics.plot(kind='bar',figsize=(15, 10),fontsize=12)
plt.tight_layout()
plt.show()
plt.close()

firstCluster = firstCluster.loc[firstCluster['NAIC-edited']=="Manufacturing"]

#   Contact-with-Objects-and-Equipment is the most common
event = firstCluster.groupby('Event-edited')['UPA'].nunique()
event
ax=event.plot(kind='bar',figsize=(15, 10),fontsize=12)
plt.tight_layout()
plt.show()
plt.close()

firstCluster = firstCluster.loc[firstCluster['Event-edited']=="Contact-with-Objects-and-Equipment"]


#   Machinery most common
source = firstCluster.groupby('Source-edited')['UPA'].nunique()
source
ax=source.plot(kind='bar',figsize=(15, 10),fontsize=12)
plt.tight_layout()
plt.show()
plt.close()

firstCluster = firstCluster.loc[firstCluster['Source-edited']=="Machinery"]


#   Upper extremities most common
body = firstCluster.groupby('Body-edited')['UPA'].nunique()
body
ax=body.plot(kind='bar',figsize=(15, 10),fontsize=12)
plt.tight_layout()
plt.show()
plt.close()
firstCluster = firstCluster.loc[firstCluster['Body-edited']=="Upper-Extremities"]


#   Verify the length
len(firstCluster)
firstCluster.dtypes

firstCluster.drop(['Primary NAICS','Hospitalized','Amputation','Secondary Source Title',
                   'NAIC-edited','Nature-edited','Event-edited','Source-edited','Body-edited'],axis=1,inplace=True)

#   Verify that there is no NA
firstCluster.isna().sum()


#   Drop replaced columns
firstCluster.drop(["NatureTitle","EventTitle","SourceTitle","Part of Body Title"],axis=1,inplace=True)

firstCluster.dtypes

#   Exporting to analyse in SAS
exportingCSV(firstCluster,"firstCluster")

#####   Moving on to our next cluster

#   Looking at all the cases
secondCluster=data.copy()

#   Verify the length
len(secondCluster)
secondCluster.dtypes

secondCluster.drop(['Primary NAICS','Hospitalized','Secondary Source Title',
                   'NAIC-edited','Nature-edited','Event-edited','Source-edited','Body-edited'],axis=1,inplace=True)

#   Verify that there is no NA
secondCluster.isna().sum()
#   NA rows insignificant, resort to dropping them
secondCluster=secondCluster.dropna()


#   Drop replaced columns
secondCluster.drop(["NatureTitle","EventTitle","SourceTitle","Part of Body Title"],axis=1,inplace=True)

secondCluster.dtypes

exportingCSV(secondCluster,"secondCluster")

####   Moving on to our next cluster

#   Looking at the cases who are hospitalised
thirdCluster = data.loc[(data['Hospitalized']=='Hospitalized')]

#   Verify the length
len(thirdCluster)
thirdCluster.dtypes

naics = thirdCluster.groupby('NAIC-edited')['UPA'].nunique()
naics
ax=naics.plot(kind='bar',figsize=(15, 10),fontsize=12)
plt.tight_layout()
plt.show()
plt.close()

thirdCluster = thirdCluster.loc[(thirdCluster['NAIC-edited']=="Administrative-and-Support-and-Waste-Management-and-Remediation-Services")|
                              (thirdCluster['NAIC-edited']=="Retail-Trade")|
                              (thirdCluster['NAIC-edited']=="Health-Care-and-Social-Assistance")|
                              (thirdCluster['NAIC-edited']=="Wholesale-Trade")]

#   Fairly well spreaded out thus resort to looking at all events
event = thirdCluster.groupby('Event-edited')['UPA'].nunique()
event

thirdCluster.drop(['Primary NAICS','Hospitalized','Secondary Source Title',
                   'NAIC-edited','Nature-edited','Event-edited','Source-edited','Body-edited'],axis=1,inplace=True)


#   Verify that there is no NA
thirdCluster.isna().sum()

#   Drop replaced columns
thirdCluster.drop(["NatureTitle","EventTitle","SourceTitle","Part of Body Title"],axis=1,inplace=True)

thirdCluster.dtypes

exportingCSV(thirdCluster,"thirdCluster")
