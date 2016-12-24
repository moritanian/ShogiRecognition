#include <ESP8266WiFi.h>
#include <WiFiClient.h>

const int pwmPin = 4; //IO4（Cerevoのブレイクアウトボードでは10番ピン）
const int swPin = 16;
const int motorPin = 13;
const int ledPin = 12;

const char SSID[] = "wx02-e4f52a";
const char PASS[] = "7f8f37091d9ae"; 

const char HOST[] = "192.168.179.7";
const int PORT = 9999;

//const long PWM_frq = 705000L;//
const long PWM_frq = 24000L;// 8000 * 256 = 2088000
const int BIT_width = 10;

const long FCY = 80000000L;

const String init_cmd = "0000";
const String config_cmd = "0001";
const String close_cmd = "0010";
const String ask_cmd = "1000";
const String kihu_cmd = "1001";
const String voice_start_cmd = "1010";
const String voice_go_cmd = "1011";
const String voice_stop_cmd = "1100";
const String voice_test_cmd = "1101"; // 特定の周波数を出力

const int buff_size = 1024;
unsigned char buff[2][buff_size];
int buff_chan = 0;
int buff_index = 0;
char buff_flg = 0;
int pwm_count = 0;
int pwm_state = 0;

unsigned int remain_channels = 0;

WiFiClient client;

//初期化処理
void setup() {
  //ESP.wdtEnable(15000);
 // ESP.wdtDisable();
  Serial.begin(115200);
  delay(10);

  pinMode(pwmPin, INPUT);
  pinMode(swPin, INPUT);
  pinMode(motorPin, OUTPUT);
  pinMode(ledPin, OUTPUT);

  //digitalWrite(pwmPin, LOW); 
  digitalWrite(ledPin, LOW);
  digitalWrite(motorPin, LOW); 
 
  noInterrupts();

  Serial.println("WiFi module setting... ");
  
  //WiFi.softAP(SSID, PASS); //任意のSSIDとPASSを指定
  WiFi.disconnect();
  WiFi.mode(WIFI_STA);
  WiFi.begin(SSID ,PASS);

  
  while (WiFi.status() != WL_CONNECTED) {
    delay(100);
    Serial.print(".");
  }
  Serial.print("IP address: ");
  Serial.println(WiFi.softAPIP());
}



void TCP_start(void){
  while (!client.connect(HOST, PORT)) {//TCPサーバへの接続要求
    Serial.print(".");
  }
}

void send_request(String line){
  client.print(line);//データを送信
}

String get_request(){
  String line;
  while(!client.available()){}
  while(client.available())
  {
    line = client.readStringUntil('\n');//受信します。
    Serial.print(line+"\r\n");
  }
  return line;
}

void get_voice_data(signed int get_chanel){
  if(get_chanel == -1){
    get_chanel = 1 - buff_chan;
  }
  int count = 0;
  int over = 0;

  send_request(voice_go_cmd);

  // 返信あるまで待機
  while(!client.available()){}

  while(count < buff_size)
  {
    while(client.available() ){
      if(count == buff_size){
        count --;
        over ++;
        //Serial.println("e1");
        //break;
      }
      buff[get_chanel][count] = (unsigned char)client.read(); // 1byteずつよみとり
      count ++;
    }
  }
  if(over){
    Serial.println(String(over));
  }else if(count < buff_size){
    Serial.println(String(count));
  }
}


// TCP通信
void TCP_client(void){ 
  WiFiClient client;
  while (!client.connect(HOST, PORT)) {//TCPサーバへの接続要求
    Serial.print(".");
  }
  
  client.print("go");//データを送信
   
  delay(10);
  // Read all the lines of the reply from server and print them to Serial
  while(client.available())
  {
    String line = client.readStringUntil('\n');//受信します。
    Serial.print(line+"\r\n");
  }
}

void timer0_ISR (void) {
  if(pwm_state == 0){
    // renown
    pwm_count++;
    if(pwm_count == 3){
      pwm_count = 0;
      buff_index ++;
      if(buff_index == buff_size){  // 再生面変更 データ取り込み
        if(remain_channels == 0){ // 終了
          Serial.println("stop timer");
          pinMode(pwmPin, INPUT);
          return;
        }
        if(buff_flg){ // まだデータとりきれてない
          Serial.println("e3");
          Serial.println(String(remain_channels));
          buff_index --;
        }else{  //反転
          //Serial.println("e2");
          buff_flg = 1;
          buff_index = 0;
          buff_chan = 1 - buff_chan;
        }
      }
    }
    
    if( buff[buff_chan][buff_index] ==0){
      digitalWrite(pwmPin, LOW);
      timer0_write(ESP.getCycleCount() + FCY/PWM_frq); // 80MHz == 1sec
    }else{
      timer0_write(ESP.getCycleCount() + FCY*buff[buff_chan][buff_index]/PWM_frq/256); // 80MHz == 1sec
      digitalWrite(pwmPin, HIGH);
      pwm_state = 1;
    }
  }else{ //state 1
     timer0_write(ESP.getCycleCount() + FCY*(256-buff[buff_chan][buff_index])/PWM_frq/256); // 80MHz == 1sec
     digitalWrite(pwmPin, LOW);
     pwm_state = 0;
  }
  //Serial.println(String(pwm_count));
}


void start_voice(){
  pinMode(pwmPin, OUTPUT);
  get_voice_data(0);
  get_voice_data(1);
  timer0_isr_init();
  timer0_attachInterrupt(timer0_ISR);
  timer0_write(ESP.getCycleCount() + 80000000L/PWM_frq); // 80MHz == 1sec
  pwm_count = 0;
  pwm_state = 0;

  buff_chan = 0;
  buff_index = 0;
  buff_flg = 0;
  
  interrupts();
 // analogWriteFreq(PWM_frq) ;
}


//本体処理
void loop() {
  int voice_size = 0;
  int line_len = 0;
  TCP_start();
  send_request(init_cmd);
  String line = get_request();
  while(1){
    delay(10);
    if(digitalRead(swPin) == HIGH){
      digitalWrite(ledPin, HIGH);
      send_request(ask_cmd);
      digitalWrite(ledPin, LOW);
      String line = get_request();
      if(line.substring(0,4).equals(kihu_cmd)){ //棋譜
        if(line.charAt(4) == '1'){ //bibe
          digitalWrite(motorPin, HIGH);
          delay(100);
          digitalWrite(motorPin, LOW);
        } 
      }else if(line.substring(0,4).equals(voice_start_cmd)){// 音声
        Serial.println("voice ");
        line_len = line.length();
        Serial.println(String(line_len));
        String voice_size_str = line.substring(4, line_len);
        Serial.println(voice_size_str);
        //voice_size = line.substring(4, line_len).toInt();
        voice_size = voice_size_str.toInt();
        Serial.println("voice size = ");
        Serial.println(String(voice_size));
        remain_channels = voice_size/buff_size - 2;
        
        Serial.println("remain channels = ");
        Serial.println(String(remain_channels));
       
        start_voice();
        Serial.println("started voice ");
    
        while(remain_channels > 0){
          if(buff_flg){
            remain_channels -= 1;
            //Serial.println("svg");
            get_voice_data(-1);
            buff_flg = 0;
            // Serial.println("gvd");
            
          }
          delay(10);
        }
        send_request(voice_stop_cmd); //voice end
      }else if(line.substring(0,4).equals(voice_test_cmd)){
        line_len = line.length();
        int f = line.substring(4, line_len).toInt();
        Serial.print("voice test");
        Serial.println(String(f));
        if(f == 0){
          digitalWrite(pwmPin, LOW); 
        }else{
          analogWrite(pwmPin, 512) ;
          analogWriteFreq(f) ;
        }
      }
    }else{
      digitalWrite(ledPin, LOW);    
    }
  }

}