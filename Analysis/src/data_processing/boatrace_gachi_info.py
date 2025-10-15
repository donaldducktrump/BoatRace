import requests
from bs4 import BeautifulSoup
from math import floor
import math

# 提供いただいたHTMLコードを文字列として保存
html_content = '''
<!doctype html>
<html lang="ja">

<head>
  <!-- Google AdSense -->
  <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-4302785053687204" data-overlays="bottom" crossorigin="anonymous"></script>
  <!-- Google tag (gtag.js) -->
  <script async src="https://www.googletagmanager.com/gtag/js?id=G-0K2LLB73YJ"></script>
  <script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', 'G-0K2LLB73YJ');
  </script>
  <!-- CMP -->
  <!-- InMobi Choice. Consent Manager Tag v3.0 (for TCF 2.2) -->
<script type="text/javascript" async=true>
(function() {
  var host = "www.themoneytizer.com";
  var element = document.createElement('script');
  var firstScript = document.getElementsByTagName('script')[0];
  var url = 'https://cmp.inmobi.com'
    .concat('/choice/', '6Fv0cGNfc_bw8', '/', host, '/choice.js?tag_version=V3');
  var uspTries = 0;
  var uspTriesLimit = 3;
  element.async = true;
  element.type = 'text/javascript';
  element.src = url;

  firstScript.parentNode.insertBefore(element, firstScript);

  function makeStub() {
    var TCF_LOCATOR_NAME = '__tcfapiLocator';
    var queue = [];
    var win = window;
    var cmpFrame;

    function addFrame() {
      var doc = win.document;
      var otherCMP = !!(win.frames[TCF_LOCATOR_NAME]);

      if (!otherCMP) {
        if (doc.body) {
          var iframe = doc.createElement('iframe');

          iframe.style.cssText = 'display:none';
          iframe.name = TCF_LOCATOR_NAME;
          doc.body.appendChild(iframe);
        } else {
          setTimeout(addFrame, 5);
        }
      }
      return !otherCMP;
    }

    function tcfAPIHandler() {
      var gdprApplies;
      var args = arguments;

      if (!args.length) {
        return queue;
      } else if (args[0] === 'setGdprApplies') {
        if (
          args.length > 3 &&
          args[2] === 2 &&
          typeof args[3] === 'boolean'
        ) {
          gdprApplies = args[3];
          if (typeof args[2] === 'function') {
            args[2]('set', true);
          }
        }
      } else if (args[0] === 'ping') {
        var retr = {
          gdprApplies: gdprApplies,
          cmpLoaded: false,
          cmpStatus: 'stub'
        };

        if (typeof args[2] === 'function') {
          args[2](retr);
        }
      } else {
        if(args[0] === 'init' && typeof args[3] === 'object') {
          args[3] = Object.assign(args[3], { tag_version: 'V3' });
        }
        queue.push(args);
      }
    }

    function postMessageEventHandler(event) {
      var msgIsString = typeof event.data === 'string';
      var json = {};

      try {
        if (msgIsString) {
          json = JSON.parse(event.data);
        } else {
          json = event.data;
        }
      } catch (ignore) {}

      var payload = json.__tcfapiCall;

      if (payload) {
        window.__tcfapi(
          payload.command,
          payload.version,
          function(retValue, success) {
            var returnMsg = {
              __tcfapiReturn: {
                returnValue: retValue,
                success: success,
                callId: payload.callId
              }
            };
            if (msgIsString) {
              returnMsg = JSON.stringify(returnMsg);
            }
            if (event && event.source && event.source.postMessage) {
              event.source.postMessage(returnMsg, '*');
            }
          },
          payload.parameter
        );
      }
    }

    while (win) {
      try {
        if (win.frames[TCF_LOCATOR_NAME]) {
          cmpFrame = win;
          break;
        }
      } catch (ignore) {}

      if (win === window.top) {
        break;
      }
      win = win.parent;
    }
    if (!cmpFrame) {
      addFrame();
      win.__tcfapi = tcfAPIHandler;
      win.addEventListener('message', postMessageEventHandler, false);
    }
  };

  makeStub();

  var uspStubFunction = function() {
    var arg = arguments;
    if (typeof window.__uspapi !== uspStubFunction) {
      setTimeout(function() {
        if (typeof window.__uspapi !== 'undefined') {
          window.__uspapi.apply(window.__uspapi, arg);
        }
      }, 500);
    }
  };

  var checkIfUspIsReady = function() {
    uspTries++;
    if (window.__uspapi === uspStubFunction && uspTries < uspTriesLimit) {
      console.warn('USP is not accessible');
    } else {
      clearInterval(uspInterval);
    }
  };

  if (typeof window.__uspapi === 'undefined') {
    window.__uspapi = uspStubFunction;
    var uspInterval = setInterval(checkIfUspIsReady, 6000);
  }
})();
</script>
<!-- End InMobi Choice. Consent Manager Tag v3.0 (for TCF 2.2) -->
<!-- Moneytizer Footer -->
  <div id="122790-6"><script src="//ads.themoneytizer.com/s/gen.js?type=6"></script><script src="//ads.themoneytizer.com/s/requestform.js?siteId=122790&formatId=6" ></script></div>
<!-- Moneytizer Footer -->
  <title>競艇予想AI ポセイドン | 本日の注目レース</title>
  <meta name="description" content="過去100万以上のレースデータをもとに作られた競艇予想の人工知能（AI）ポセイドンが、レースを予想します。全ての予想を完全無料でご利用いただけます。">
  <meta name="theme-color" content="#1667d9">
  <meta charset="utf-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="manifest" href="/manifest.json">
  <link rel="apple-touch-icon" href="/images/icon/apple-touch-icon.png">
  <link href="/css/bootstrap.css?20210723" rel="stylesheet" type="text/css">
  <link href="/css/main.css?20210723" rel="stylesheet" type="text/css">
  <link href="/css/iconic/css/open-iconic-bootstrap.css" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap-material-datetimepicker/2.7.1/css/bootstrap-material-datetimepicker.min.css">
  <link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons">
</head>

<body>
  <header>
    <nav class="navbar navbar-expand-lg bg-primary py-0 pr-0">
      <a class="navbar-brand" href="/">
        <img src="/images/logo.svg" class="site-logo" height="40" alt="logo">
        <img src="/images/wordlogo.svg" class="site-logo-word" height="26" alt="wordlogo">
      </a>
      <div class="collapse navbar-collapse bg-background">
        <ul class="navbar-nav mr-auto">
          <li class="nav-item px-1">
            <a class="nav-link text-primary font-weight-bold px-4 pt-3 border-bottom-nav" href="/race">
              <svg id="flag" viewBox="0 0 512 512" style="width: 20px; height: 20px;">
                <g>
                	<path class="nav-logo" d="M223.546,83.775c59.344,10.281,195.891,87.547,177.641,144l-5.219,24.469c0,0,24.719,73.516,116.031,70.938
                		c-18.266,59.016-68.375,76.859-129.734,80.422c-92.688-5.328-80.125-31.844-80.125-31.844s76.469-40.625-160.766-67.328
                		L223.546,83.775z"></path>
                	<path class="nav-logo" d="M24.031,486.275c-2.797,0-5.625-0.484-8.391-1.516c-12.422-4.625-18.75-18.453-14.125-30.891L155.156,41.354
                		c4.641-12.438,18.5-18.719,30.875-14.125c12.438,4.625,18.766,18.453,14.125,30.891L46.531,470.635
                		C42.937,480.307,33.765,486.275,24.031,486.275z"></path>
                </g>
              </svg>
              レース予想
            </a>
          </li>
          <li class="nav-item px-1">
            <a class="nav-link text-primary font-weight-bold px-4 pt-3 border-bottom-nav active" href="/pickup">
              <svg id="pickup" viewBox="0 0 512 512" style="width: 20px; height: 20px;">
                <g>
                	<path class="nav-logo" d="M404.563,188.938c-23.078-36.063-38.953-60.578-64.906-121.156c0,0-18.75,24.531-33.172,60.578
                		c-6.703-23.203-18.922-33.578-33.172-57.703C254.563,38.938,251.672,0,251.672,0s-30.281,17.313-50.484,60.563
                		c-20.172,43.281-18.75,93.75-41.813,116.844c-8.656-46.156-30.297-59.125-30.297-59.125s-12.984,38.906-21.625,64.875
                		c-23.078,46.156-37.5,91.406-37.5,142.797c0,51.359,20.813,97.891,54.5,131.547C158.125,491.188,204.641,512,256,512
                		c51.375,0,97.891-20.813,131.563-54.5c33.672-33.656,54.484-80.188,54.484-131.547
                		C442.047,274.563,424.766,223.547,404.563,188.938z M336.563,438.25c-21.516,21.5-50.125,33.375-80.563,33.375
                		s-59.047-11.875-80.563-33.375c-21.531-21.531-33.375-50.141-33.375-80.563c0-10.656,0.844-21.188,2.594-31.875
                		c6.063-24.375,15.688-59,15.688-59s24.406,22.156,28.828,36.531c7.703,25,19.234-15.375,29.328-37.969
                		c8.656-10.094,18.625-42.344,25.297-62.5l59.797,108.656l40.5-51.328c12.813,23.219,25.859,60,25.859,97.484
                		C369.953,388.109,358.078,416.719,336.563,438.25z"></path>
                </g>
              </svg>
              注目レース
            </a>
          </li>
          <li class="nav-item px-1">
            <a class="nav-link text-primary font-weight-bold px-4 pt-3 border-bottom-nav" href="/history">
              <svg id="history" viewBox="0 0 512 512" style="width: 20px; height: 20px;">
                <g>
                  <path class="nav-logo" d="M204.762,254.456l34.212-34.204c-39.807-18.293-88.544-11.079-121.29,21.675
                		c-42.013,42.006-42.013,110.372,0,152.393c42.005,42.014,110.38,42.014,152.386,0c32.746-32.745,39.968-81.49,21.675-121.298
                		l-34.211,34.211c3.381,19.976-2.553,41.224-17.939,56.604c-25.21,25.218-66.225,25.218-91.434,0
                		c-25.21-25.21-25.21-66.224,0-91.427C163.546,257.016,184.794,251.074,204.762,254.456z"></path>
                	<path class="nav-logo" d="M323.628,241.146c34.324,57.876,26.642,133.939-23.076,183.65c-58.826,58.826-154.527,58.826-213.345,0
                		c-58.826-58.817-58.826-154.527,0-213.352c49.703-49.711,125.775-57.393,183.65-23.076l31.216-31.225
                		c-75.387-50.693-178.754-42.77-245.35,23.817c-75.629,75.621-75.629,198.69,0,274.311c75.63,75.638,198.683,75.638,274.312,0
                		c66.603-66.595,74.518-169.962,23.809-245.358L323.628,241.146z"></path>
                	<path class="nav-logo" d="M511.279,84.84c-1.61-4.195-5.684-6.78-10.298-6.57l-70.565,3.31l3.318-70.556
                		c0.201-4.622-2.384-8.68-6.578-10.306c-4.17-1.61-9.122-0.451-12.52,2.931l-75.299,75.306l-3.809,81.322L198.634,297.162
                		c-6.964-1.578-14.565,0.29-19.992,5.716c-8.422,8.422-8.422,22.062,0,30.484c8.414,8.422,22.062,8.422,30.484,0
                		c5.418-5.427,7.295-13.028,5.716-20l136.886-136.894l81.314-3.8l75.307-75.316C511.739,93.963,512.89,89.026,511.279,84.84z"></path>
                </g>
              </svg>
              的中結果
            </a>
          </li>
        </ul>
        <span class="navbar-text bg-background pr-5">
        </span>
      </div>
    </nav>
  </header>

  <div class="row sp-nav bg-primary mx-0">
    <div class="col-4">
      <a class="nav-link px-4 sph text-center" href="/race">
        <svg id="flag" viewBox="0 0 512 512" style="width: 16px; height: 16px;">
          <g>
            <path class="sph-nav-logo" d="M223.546,83.775c59.344,10.281,195.891,87.547,177.641,144l-5.219,24.469c0,0,24.719,73.516,116.031,70.938
              c-18.266,59.016-68.375,76.859-129.734,80.422c-92.688-5.328-80.125-31.844-80.125-31.844s76.469-40.625-160.766-67.328
              L223.546,83.775z"></path>
            <path class="sph-nav-logo" d="M24.031,486.275c-2.797,0-5.625-0.484-8.391-1.516c-12.422-4.625-18.75-18.453-14.125-30.891L155.156,41.354
              c4.641-12.438,18.5-18.719,30.875-14.125c12.438,4.625,18.766,18.453,14.125,30.891L46.531,470.635
              C42.937,480.307,33.765,486.275,24.031,486.275z"></path>
          </g>
        </svg><br>
        <small class="font-weight-bold">レース予想</small>
      </a>
    </div>
    <div class="col-4">
      <a class="nav-link px-4 sph text-center active" href="/pickup">
        <svg id="pickup" viewBox="0 0 512 512" style="width: 16px; height: 16px;">
          <g>
            <path class="sph-nav-logo active" d="M404.563,188.938c-23.078-36.063-38.953-60.578-64.906-121.156c0,0-18.75,24.531-33.172,60.578
          		c-6.703-23.203-18.922-33.578-33.172-57.703C254.563,38.938,251.672,0,251.672,0s-30.281,17.313-50.484,60.563
          		c-20.172,43.281-18.75,93.75-41.813,116.844c-8.656-46.156-30.297-59.125-30.297-59.125s-12.984,38.906-21.625,64.875
          		c-23.078,46.156-37.5,91.406-37.5,142.797c0,51.359,20.813,97.891,54.5,131.547C158.125,491.188,204.641,512,256,512
          		c51.375,0,97.891-20.813,131.563-54.5c33.672-33.656,54.484-80.188,54.484-131.547
          		C442.047,274.563,424.766,223.547,404.563,188.938z M336.563,438.25c-21.516,21.5-50.125,33.375-80.563,33.375
          		s-59.047-11.875-80.563-33.375c-21.531-21.531-33.375-50.141-33.375-80.563c0-10.656,0.844-21.188,2.594-31.875
          		c6.063-24.375,15.688-59,15.688-59s24.406,22.156,28.828,36.531c7.703,25,19.234-15.375,29.328-37.969
          		c8.656-10.094,18.625-42.344,25.297-62.5l59.797,108.656l40.5-51.328c12.813,23.219,25.859,60,25.859,97.484
          		C369.953,388.109,358.078,416.719,336.563,438.25z"></path>
          </g>
        </svg><br>
        <small class="font-weight-bold">注目レース</small>
      </a>
    </div>
    <div class="col-4">
      <a class="nav-link px-4 sph text-center" href="/history">
        <svg id="history" viewBox="0 0 512 512" style="width: 16px; height: 16px;">
          <g>
            <path class="sph-nav-logo" d="M204.762,254.456l34.212-34.204c-39.807-18.293-88.544-11.079-121.29,21.675
              c-42.013,42.006-42.013,110.372,0,152.393c42.005,42.014,110.38,42.014,152.386,0c32.746-32.745,39.968-81.49,21.675-121.298
              l-34.211,34.211c3.381,19.976-2.553,41.224-17.939,56.604c-25.21,25.218-66.225,25.218-91.434,0
              c-25.21-25.21-25.21-66.224,0-91.427C163.546,257.016,184.794,251.074,204.762,254.456z"></path>
            <path class="sph-nav-logo" d="M323.628,241.146c34.324,57.876,26.642,133.939-23.076,183.65c-58.826,58.826-154.527,58.826-213.345,0
              c-58.826-58.817-58.826-154.527,0-213.352c49.703-49.711,125.775-57.393,183.65-23.076l31.216-31.225
              c-75.387-50.693-178.754-42.77-245.35,23.817c-75.629,75.621-75.629,198.69,0,274.311c75.63,75.638,198.683,75.638,274.312,0
              c66.603-66.595,74.518-169.962,23.809-245.358L323.628,241.146z"></path>
            <path class="sph-nav-logo" d="M511.279,84.84c-1.61-4.195-5.684-6.78-10.298-6.57l-70.565,3.31l3.318-70.556
              c0.201-4.622-2.384-8.68-6.578-10.306c-4.17-1.61-9.122-0.451-12.52,2.931l-75.299,75.306l-3.809,81.322L198.634,297.162
              c-6.964-1.578-14.565,0.29-19.992,5.716c-8.422,8.422-8.422,22.062,0,30.484c8.414,8.422,22.062,8.422,30.484,0
              c5.418-5.427,7.295-13.028,5.716-20l136.886-136.894l81.314-3.8l75.307-75.316C511.739,93.963,512.89,89.026,511.279,84.84z"></path>
          </g>
        </svg><br>
        <small class="font-weight-bold">的中結果</small>
      </a>
    </div>
  </div>
  <main role="main" class="container">
        <!--
      <nav class="nav nav-pills nav-justified bg-white shadow-sm nav-border border-primary my-2">
        <a class="nav-item nav-link fontsmall-sm font-weight-bold" href="/">トップ</a>
        <a class="nav-item nav-link fontsmall-sm font-weight-bold" href="/race">レース予想</a>
        <a class="nav-item nav-link fontsmall-sm font-weight-bold active" href="/pickup">注目レース</a>
        <a class="nav-item nav-link fontsmall-sm font-weight-bold" href="/history">的中結果</a>
      </nav>
    -->
        <div class="row">
  <div class="col-12 py-2">
    <div class="card bg-white shadow-sm ticket-border border-secondary pb-2">
      <!-- ポセイドン-ディスプレイ広告 -->
      <ins class="adsbygoogle"
           style="display:block"
           data-ad-client="ca-pub-4302785053687204"
           data-ad-slot="8001267344"
           data-ad-format="auto"
           data-full-width-responsive="true"></ins>
      <script>
           (adsbygoogle = window.adsbygoogle || []).push({});
      </script>
      <div class="card-header bg-white">
        <h3 class="h4 mb-0">本日の注目レース</h3>
      </div>

      <div class="card-body">
        <!-- タブボタン -->
        <ul class="nav nav-tabs row">
          <li class="nav-item col-6 detail-item">
            <a href="#nigeru" class="nav-link font-weight-bold text-center" onClick="gtag('event','click',{'event_category':'pickup','event_label':'nigeru'});" data-toggle="tab">ガチガチレース</a>
          </li>
          <li class="nav-item col-6 detail-item">
            <a href="#nigasan" class="nav-link font-weight-bold text-center" onClick="gtag('event','click',{'event_category':'pickup','event_label':'nigasan'});" data-toggle="tab">穴レース</a>
          </li>
        </ul>

        <div class="tab-content">
          <!-- デフォルト -->
          <div id="default" class="tab-pane row active">
            <table class="table table-borderless">
              <tr>
                <th width="50%" scope="col" class="text-center pb-0"><span class="oi oi-arrow-thick-top h2"></span></th>
                <th width="50%" scope="col" class="text-center pb-0"><span class="oi oi-arrow-thick-top h2"></span></th>
              </tr>
              <tr>
                <td class="table-fontsmall-sm">
                  <span class="font-weight-bold">イン逃げ(1号艇が1着)</span>の確率が高いレースのランキングです。<br>
                  ガチガチなレース展開が予想されるので、狙い目の絞り込みなどにお使いください。
                </td>
                <td class="table-fontsmall-sm">
                  <span class="font-weight-bold">2～6号艇が1着</span>の確率が高いレースのランキングです。
                  差し・まくりの展開で荒れる可能性があります。<br>
                  <span class="font-weight-bold">中穴～大穴狙い</span>にお使いください。
                </td>
              </tr>
            </table>
          </div>

          <!-- ガチガチ -->
          <div id="nigeru" class="tab-pane row">
            <table class="table table-fontsmall-sm">
              <thead>
                <tr>
                  <th width="30%" scope="col" class="text-center">レース</th>
                  <th width="25%" scope="col" class="text-center">AI予想確率</th>
                  <th width="20%" colspan="2" scope="col" class="text-center">結果</th>
                  <th width="25%" scope="col" class="text-center">配当</th>
                </tr>
              </thead>
              <tbody>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/19/8R">下関8R</a></td>
                  <td class="text-center">87.36%</td>
                                      <td class="text-center pr-0">
                      1-4-3
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      1,100円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/9/5R">津5R</a></td>
                  <td class="text-center">85.19%</td>
                                      <td class="text-center pr-0">
                      1-6-3
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      2,900円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/9/1R">津1R</a></td>
                  <td class="text-center">83.19%</td>
                                      <td class="text-center pr-0">
                      1-2-3
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      1,300円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/22/7R">福岡7R</a></td>
                  <td class="text-center">82.56%</td>
                                      <td class="text-center pr-0">
                      1-2-4
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      580円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/19/12R">下関12R</a></td>
                  <td class="text-center">80.30%</td>
                                      <td class="text-center pr-0">
                      1-6-3
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      12,530円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/23/10R">唐津10R</a></td>
                  <td class="text-center">79.81%</td>
                                      <td class="text-center pr-0">
                      1-2-3
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      1,080円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/7/11R">蒲郡11R</a></td>
                  <td class="text-center">79.55%</td>
                                      <td class="text-center pr-0">
                      1-6-4
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      2,570円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/10/3R">三国3R</a></td>
                  <td class="text-center">78.38%</td>
                                      <td class="text-center pr-0">
                      1-4-3
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      1,320円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/24/12R">大村12R</a></td>
                  <td class="text-center">77.50%</td>
                                      <td class="text-center pr-0">
                      4-5-1
                    </td>
                    <td class="text-center px-0">
                                          </td>
                  
                                      <td class="text-center">
                      12,690円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/23/12R">唐津12R</a></td>
                  <td class="text-center">77.45%</td>
                                      <td class="text-center pr-0">
                      1-2-3
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      520円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/4/12R">平和島12R</a></td>
                  <td class="text-center">77.40%</td>
                                      <td class="text-center pr-0">
                      2-1-6
                    </td>
                    <td class="text-center px-0">
                                          </td>
                  
                                      <td class="text-center">
                      3,690円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/24/6R">大村6R</a></td>
                  <td class="text-center">77.36%</td>
                                      <td class="text-center pr-0">
                      1-4-3
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      1,470円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/24/4R">大村4R</a></td>
                  <td class="text-center">77.21%</td>
                                      <td class="text-center pr-0">
                      1-6-5
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      7,470円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/23/3R">唐津3R</a></td>
                  <td class="text-center">76.41%</td>
                                      <td class="text-center pr-0">
                      1-3-4
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      1,110円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/24/11R">大村11R</a></td>
                  <td class="text-center">75.84%</td>
                                      <td class="text-center pr-0">
                      1-2-3
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      410円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/19/9R">下関9R</a></td>
                  <td class="text-center">75.27%</td>
                                      <td class="text-center pr-0">
                      5-1-3
                    </td>
                    <td class="text-center px-0">
                                          </td>
                  
                                      <td class="text-center">
                      4,570円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/24/5R">大村5R</a></td>
                  <td class="text-center">75.19%</td>
                                      <td class="text-center pr-0">
                      3-1-4
                    </td>
                    <td class="text-center px-0">
                                          </td>
                  
                                      <td class="text-center">
                      20,670円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/4/5R">平和島5R</a></td>
                  <td class="text-center">74.85%</td>
                                      <td class="text-center pr-0">
                      1-2-6
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      4,260円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/7/10R">蒲郡10R</a></td>
                  <td class="text-center">74.75%</td>
                                      <td class="text-center pr-0">
                      1-2-6
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      780円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/24/1R">大村1R</a></td>
                  <td class="text-center">74.36%</td>
                                      <td class="text-center pr-0">
                      1-3-6
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      2,460円
                    </td>
                                  </tr>
                              </tbody>
            </table>
          </div>

          <!-- 穴 -->
          <div id="nigasan" class="tab-pane row">
            <table class="table table-fontsmall-sm">
              <thead>
                <tr>
                  <th width="30%" scope="col" class="text-center">レース</th>
                  <th width="25%" scope="col" class="text-center">AI予想確率</th>
                  <th width="20%" colspan="2" scope="col" class="text-center">結果</th>
                  <th width="25%" scope="col" class="text-center">配当</th>
                </tr>
              </thead>
              <tbody>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/9/6R">津6R</a></td>
                  <td class="text-center">91.41%</td>
                                      <td class="text-center pr-0">
                      2-1-3
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      700円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/3/7R">江戸川7R</a></td>
                  <td class="text-center">89.26%</td>
                                      <td class="text-center pr-0">
                      2-1-3
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      9,050円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/22/5R">福岡5R</a></td>
                  <td class="text-center">86.15%</td>
                                      <td class="text-center pr-0">
                      6-3-4
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      8,910円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/4/7R">平和島7R</a></td>
                  <td class="text-center">85.30%</td>
                                      <td class="text-center pr-0">
                      4-3-6
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      4,530円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/4/4R">平和島4R</a></td>
                  <td class="text-center">83.82%</td>
                                      <td class="text-center pr-0">
                      1-3-5
                    </td>
                    <td class="text-center px-0">
                                          </td>
                  
                                      <td class="text-center">
                      1,540円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/23/5R">唐津5R</a></td>
                  <td class="text-center">78.70%</td>
                                      <td class="text-center pr-0">
                      3-1-5
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      1,440円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/10/8R">三国8R</a></td>
                  <td class="text-center">78.17%</td>
                                      <td class="text-center pr-0">
                      1-2-3
                    </td>
                    <td class="text-center px-0">
                                          </td>
                  
                                      <td class="text-center">
                      6,930円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/9/4R">津4R</a></td>
                  <td class="text-center">77.72%</td>
                                      <td class="text-center pr-0">
                      6-1-5
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      32,110円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/23/6R">唐津6R</a></td>
                  <td class="text-center">74.93%</td>
                                      <td class="text-center pr-0">
                      1-3-6
                    </td>
                    <td class="text-center px-0">
                                          </td>
                  
                                      <td class="text-center">
                      1,630円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/2/9R">戸田9R</a></td>
                  <td class="text-center">73.69%</td>
                                      <td class="text-center pr-0">
                      3-5-6
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      2,540円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/4/3R">平和島3R</a></td>
                  <td class="text-center">71.08%</td>
                                      <td class="text-center pr-0">
                      1-4-5
                    </td>
                    <td class="text-center px-0">
                                          </td>
                  
                                      <td class="text-center">
                      7,480円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/19/1R">下関1R</a></td>
                  <td class="text-center">70.05%</td>
                                      <td class="text-center pr-0">
                      6-1-4
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      71,270円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/23/7R">唐津7R</a></td>
                  <td class="text-center">69.96%</td>
                                      <td class="text-center pr-0">
                      5-4-1
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      19,330円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/7/8R">蒲郡8R</a></td>
                  <td class="text-center">69.48%</td>
                                      <td class="text-center pr-0">
                      3-1-2
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      2,270円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/4/9R">平和島9R</a></td>
                  <td class="text-center">69.42%</td>
                                      <td class="text-center pr-0">
                      4-5-2
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      38,210円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/6/8R">浜名湖8R</a></td>
                  <td class="text-center">69.05%</td>
                                      <td class="text-center pr-0">
                      5-6-1
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      42,970円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/4/10R">平和島10R</a></td>
                  <td class="text-center">68.87%</td>
                                      <td class="text-center pr-0">
                      2-5-3
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      5,380円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/10/9R">三国9R</a></td>
                  <td class="text-center">68.53%</td>
                                      <td class="text-center pr-0">
                      2-3-4
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      10,520円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/22/2R">福岡2R</a></td>
                  <td class="text-center">67.88%</td>
                                      <td class="text-center pr-0">
                      1-4-3
                    </td>
                    <td class="text-center px-0">
                                          </td>
                  
                                      <td class="text-center">
                      800円
                    </td>
                                  </tr>
                                <tr>
                  <td class="text-center"><a href="/race/20241003/6/9R">浜名湖9R</a></td>
                  <td class="text-center">67.56%</td>
                                      <td class="text-center pr-0">
                      5-2-3
                    </td>
                    <td class="text-center px-0">
                                              <span class="oi oi-target text-danger"></span>
                                          </td>
                  
                                      <td class="text-center">
                      30,350円
                    </td>
                                  </tr>
                              </tbody>
            </table>
          </div>

        </div>
      </div>
    </div>
  </div>
</div>
  </main>

  <!-- 余白 -->
  <div style="padding-bottom: 240px;"></div>
  <!-- 余白 -->

  <footer class="footer bg-info">
    <div class="container text-center">
      <span class="text-muted"><a href="https://forms.gle/eNrAo4368LXXRjHv7">お問い合わせ</a> | <a href="https://twitter.com/AI_POSEIDON">公式Twitter</a></span>
    </div>
  </footer>
  <!-- Page-top Button -->
  <div id="page_top"><a href="#"></a></div>
      <!-- Bootstrap core JavaScript -->
    <script src="https://code.jquery.com/jquery-3.3.1.min.js" integrity="sha256-FgpCb/KJQlLNfOu91ta32o/NMZxltwRo8QtmkMRdAu8=" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.3/umd/popper.min.js" integrity="sha384-ZMP7rVo3mIykV+2+9J3UJ46jBk0WLaUAdn689aCwoqbBJiSnjAK/l8WvCWPIPm49" crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/js/bootstrap.min.js" integrity="sha384-ChfqqxuZUCnJSK3+MXmPNIyE6ZbWh2IMqE241rYiqJxyMiZ6OW/JmZQ5stwEULTy" crossorigin="anonymous"></script>
  </body>

</html>

'''

# BeautifulSoupでパース
soup = BeautifulSoup(html_content, 'html.parser')

# 「ガチガチレース」のタブを持つ<div>を取得
nigeru_tab = soup.find('div', id='nigeru')

# レースのテーブルを取得
race_table = nigeru_tab.find('table', class_='table')

# テーブルの中の全ての行（trタグ）を取得
rows = race_table.find_all('tr')

race_links = []

# ヘッダー行をスキップするために、最初の行を除外
for row in rows[1:]:
    cols = row.find_all('td')
    if len(cols) >= 1:
        # レースリンクを含む<a>タグを取得
        race_link_tag = cols[0].find('a')
        if race_link_tag:
            race_name = race_link_tag.text.strip()
            race_link = race_link_tag['href']
            ai_prediction = cols[1].text.strip()
            # 結果や配当も必要なら取得可能
            race_links.append({
                'race_name': race_name,
                'race_link': race_link,
                'ai_prediction': ai_prediction
            })

print(race_links)

# 資金配分を計算する関数
def calculate_bet_allocation(predictions, total_funds, minimum_bet):
    top_10_predictions = predictions[:10]
    total_inverse_odds = sum([1/p['odds'] for p in top_10_predictions if p['odds'] != -1.0])
    bet_allocations = []
    total_bet_amount = 0

    for prediction in top_10_predictions:
        odds = prediction['odds']
        trifecta = prediction['trifecta']

        if odds == -1.0:
            bet_amount = total_funds  # 全額返金
            refund_amount = total_funds
            bet_allocations.append({
                'trifecta': trifecta,
                'bet_amount': bet_amount,
                'odds': odds,
                'refund_amount': refund_amount
            })
            return bet_allocations  # 返金がある場合、処理を終了

        allocated_fund = (total_funds / odds) / total_inverse_odds
        bet_amount = max(math.floor(allocated_fund / minimum_bet) * minimum_bet, minimum_bet)
        total_bet_amount += bet_amount

        bet_allocations.append({
            'trifecta': trifecta,
            'bet_amount': bet_amount,
            'odds': odds,
            'refund_amount': 0
        })

    remaining_funds = total_funds - total_bet_amount

    if remaining_funds > 0:
        for bet in bet_allocations:
            if remaining_funds >= minimum_bet:
                bet['bet_amount'] += minimum_bet
                remaining_funds -= minimum_bet
            if remaining_funds < minimum_bet:
                break

    return bet_allocations

# 回収率を計算する関数
def calculate_return_rate(bet_allocations, race_result):
    total_investment = sum([bet['bet_amount'] for bet in bet_allocations])
    payout = 0

    for bet in bet_allocations:
        if bet['trifecta'] == race_result:
            payout += bet['bet_amount'] * bet['odds']
            break

    return_rate = payout / total_investment if total_investment > 0 else 0
    return return_rate, payout, total_investment

# 計算結果をファイルに書き込む
def save_results_to_file(bet_allocations, return_rate, payout, total_investment, filename="bet_allocations.txt"):
    with open(filename, "w") as f:
        for bet in bet_allocations:
            f.write(f"Trifecta: {bet['trifecta']}, Bet Amount: {bet['bet_amount']}, Odds: {bet['odds']}, Refund: {bet['refund_amount']}\n")
        f.write(f"\nReturn Rate: {return_rate*100:.2f}%, Payout: {payout}, Total Investment: {total_investment}\n")



# まず、回収率100%を超えたレース番号のリストを保持するリストを作成します
profitable_races = []
total_investment = 0
total_return = 0

for race in race_links:
    race_url = 'https://poseidon-boatrace.net' + race['race_link']
    print(f"アクセス中: {race_url}")
    
    # 実際のアクセスは利用規約を確認してから行ってください
    # response = requests.get(race_url)
    # if response.status_code == 200:
    #     race_soup = BeautifulSoup(response.content, 'html.parser')
    #     # 予想情報を抽出する処理をここに記述
    #     prediction = race_soup.find(...)  # 適切なセレクタを使用
    #     race['prediction'] = prediction.text.strip()
    # else:
    #     print(f"Failed to access {race_url}")

    # ページを取得
    response = requests.get(race_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # 三連単の予想確率の部分を特定
    predictions = []

    # 例として、予想確率が<span class="probability">にある場合
    a3ren_table = soup.find('div', id='a3ren').find('table')
    # tbodyを取得
    tbody = a3ren_table.find('tbody')

    # tbody内の全ての行(tr)を取得
    rows = tbody.find_all('tr')


    # 各行からデータを抽出
    for row in rows:
        # 'th'タグが存在するか確認
        trifecta_th = row.find('th')
        if trifecta_th is not None:
            trifecta = trifecta_th.text.strip()  # 組み合わせ
            columns = row.find_all('td')
            if len(columns) >= 3:
                probability = columns[0].text.strip()  # AI予想確率
                odds = columns[1].text.strip()  # オッズ
                kaijin_index = columns[2].text.strip()  # 海神指数

                # 結果を保存
                predictions.append({
                    'trifecta': trifecta,
                    'probability': float(probability.strip('%')) / 100,  # 確率を0から1の範囲に変換
                    'odds': float(odds),
                    'kaijin_index': kaijin_index
                })

    # 抽出結果を確認
    for prediction in predictions:
        print(prediction)
    
    # 総資金
    total_funds = 10000
    minimum_bet = 100

    # 仮のレース結果（本来はスクレイピングで取得）
    race_result = "1-2-3"  # レース結果

    # 資金配分を計算
    bet_allocations = calculate_bet_allocation(predictions, total_funds, minimum_bet)

    # 回収率を計算
    return_rate, payout, total_investment = calculate_return_rate(bet_allocations, race_result)
    
    if return_rate > 100:
        profitable_races.append(race_links.index(race)+1)

    # 結果を表示
    print(f"回収率: {return_rate:.2f}%")
    print(f"投資金額: {total_investment}円")
    print(f"払い戻し金額: {payout}円")



def output_results_to_file(results, filename="bet_results.txt"):
    with open(filename, "w", encoding="utf-8") as file:
        for result in results:
            file.write(f"Race: {result['race']}\n")
            file.write(f"Trifecta: {result['trifecta']}\n")
            file.write(f"Bet Amount: {result['bet_amount']} 円\n")
            file.write(f"Odds: {result['odds']}\n")
            file.write(f"Payout: {result['payout']} 円\n")
            file.write(f"Total Investment: {result['total_investment']} 円\n")
            file.write(f"Return Rate: {result['return_rate'] * 100:.2f}%\n")
            file.write("\n")

# 結果をファイルに保存
results = [{
    'race': 'Race 1',
    'trifecta': race_result,
    'bet_amount': sum([bet['bet_amount'] for bet in bet_allocations]),
    'odds': max([bet['odds'] for bet in bet_allocations]),
    'payout': payout,
    'total_investment': total_investment,
    'return_rate': return_rate
}]

output_results_to_file(results)

print(f"回収率が100%を超えたレース番号: {profitable_races}")

