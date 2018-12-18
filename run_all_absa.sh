mkdir logs
export LOGFILE=logs/absa_outs.txt
python exps_absa.py 0 | tee $LOGFILE
python exps_absa.py 1 | tee -a $LOGFILE
python exps_absa.py 2 | tee -a $LOGFILE
python exps_absa.py 3 | tee -a $LOGFILE
python exps_absa.py 4 | tee -a $LOGFILE
python exps_absa.py 5 | tee -a $LOGFILE
python exps_absa.py 6 | tee -a $LOGFILE
python exps_absa.py 7 | tee -a $LOGFILE
python exps_absa.py 8 | tee -a $LOGFILE
python exps_absa.py 9 | tee -a $LOGFILE
python exps_absa.py 10 | tee -a $LOGFILE
python exps_absa.py 11 | tee -a $LOGFILE
python exps_absa.py 12 | tee -a $LOGFILE
python exps_absa.py 13 | tee -a $LOGFILE
python exps_absa.py 14 | tee -a $LOGFILE
python exps_absa.py 15 | tee -a $LOGFILE
python exps_absa.py 16 | tee -a $LOGFILE
python exps_absa.py 17 | tee -a $LOGFILE
