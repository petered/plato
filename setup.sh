virtualenv venv
source venv/bin/activate
printf '\nexport PYTHONPATH=$PYTHONPATH:%s\n' $PWD >> venv/bin/activate
pip install -r requirements.txt
echo “Plato Setup Complete! (As long as there’re no errors above)”