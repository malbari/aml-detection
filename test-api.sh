#!/bin/bash

#header
#step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud

#line="1,PAYMENT,9839.64,C1231006815,170136.0,160296.36,M1979787155,0.0,0.0,0,0"
#line="1,PAYMENT,1864.28,C1666544295,21249.0,19384.72,M2044282225,0.0,0.0,0,0"
line="1,TRANSFER,181.0,C1305486145,181.0,0.0,C553264065,0.0,0.0,1,0"
#line="1,CASH_OUT,181.0,C840083671,181.0,0.0,C38997010,21182.0,0.0,1,0"
#line="1,PAYMENT,11668.14,C2048537720,41554.0,29885.86,M1230701703,0.0,0.0,0,0"
#line="1,PAYMENT,7817.71,C90045638,53860.0,46042.29,M573487274,0.0,0.0,0,0"
#line="1,PAYMENT,7107.77,C154988899,183195.0,176087.23,M408069119,0.0,0.0,0,0"
#line="1,PAYMENT,7861.64,C1912850431,176087.23,168225.59,M633326333,0.0,0.0,0,0"
#line="1,PAYMENT,4024.36,C1265012928,2671.0,0.0,M1176932104,0.0,0.0,0,0"
#line="1,DEBIT,5337.77,C712410124,41720.0,36382.23,C195600860,41898.0,40348.79,0,0"
#line="1,DEBIT,9644.94,C1900366749,4465.0,0.0,C997608398,10845.0,157982.12,0,0"
#line="1,PAYMENT,3099.97,C249177573,20771.0,17671.03,M2096539129,0.0,0.0,0,0"
#line="1,PAYMENT,2560.74,C1648232591,5070.0,2509.26,M972865270,0.0,0.0,0,0"
#line="1,PAYMENT,11633.76,C1716932897,10127.0,0.0,M801569151,0.0,0.0,0,0"
#line="1,PAYMENT,4098.78,C1026483832,503264.0,499165.22,M1635378213,0.0,0.0,0,0"
# return a wrong value
#line="1,CASH_OUT,229133.94,C905080434,15325.0,0.0,C476402209,5083.0,51513.44,0,0"
#line="1,PAYMENT,1563.82,C761750706,450.0,0.0,M1731217984,0.0,0.0,0,0"
#line="1,PAYMENT,1157.86,C1237762639,21156.0,19998.14,M1877062907,0.0,0.0,0,0"
#line="1,PAYMENT,671.64,C2033524545,15123.0,14451.36,M473053293,0.0,0.0,0,0"

arr=($(echo ${line} | tr ',' '\n'))

step=${arr[0]}
type=${arr[1]}
amount=${arr[2]}
nameOrig=${arr[3]}
oldbalanceOrg=${arr[4]}
newbalanceOrig=${arr[5]}
nameDest=${arr[6]}
oldbalanceDest=${arr[7]}
newbalanceDest=${arr[8]}

URL="http://127.0.0.1:5000/predict?step=${step}&type=${type}&amount=${amount}&nameOrig=${nameOrig}&oldbalanceOrg=${oldbalanceOrg}&newbalanceOrig=${newbalanceOrig}&nameDest=${nameDest}&oldbalanceDest=${oldbalanceDest}&newbalanceDest=${newbalanceDest}"

curl $URL