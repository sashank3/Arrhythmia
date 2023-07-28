void setup() {
// initialize the serial communication:
Serial.begin(9600);
pinMode(10, INPUT); // Setup for leads off detection LO +
pinMode(11, INPUT); // Setup for leads off detection LO -
}
int counter = 0;
 
void loop() {

counter = counter+1;
Serial.println(analogRead(A0));
//Wait for a bit to keep serial data from saturating
delay(1);
}
