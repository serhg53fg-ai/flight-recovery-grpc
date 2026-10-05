import os
import threading
from pathlib import Path
import pytest
from werkzeug.serving import make_server
from tests.integration.test_prediction_flow import app, gateway_address, FLIGHT

pytestmark=pytest.mark.skipif(os.environ.get('FLIGHT_BROWSER_TESTS')!='1',reason='opt-in browser acceptance')

@pytest.fixture
def browser(app):
    from playwright.sync_api import sync_playwright
    server=make_server('127.0.0.1',0,app,threaded=True)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        context=browser.new_context(viewport={'width':1440,'height':1000},timezone_id='Asia/Shanghai')
        try: yield context,f'http://127.0.0.1:{server.server_port}'
        finally: context.close();browser.close()
    server.shutdown();thread.join();server.server_close()

def test_selection_source_and_tab_isolation(browser):
    context,base=browser
    a=context.new_page();b=context.new_page()
    a.goto(base+'/flight_input');b.goto(base+'/flight_input')
    def predict(page,number):
        return page.evaluate('''async flight => (await fetch('/predict_flight', {
          method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(flight)})).json()''',
          {**FLIGHT,'航班号':number})
    ar=predict(a,'CZ1111');br=predict(b,'CZ2222')
    assert ar['job_id']!=br['job_id']
    from playwright.sync_api import expect
    expect(a.locator('#prediction-source')).to_contain_text('TEST')
    a.locator('a[href^="/timeline"]').first.click()
    assert ar['job_id'] in a.url
    a.reload()
    result=a.evaluate("async () => (await fetch('/api/flight-data')).json()")
    assert result['job_id']==ar['job_id'] and result['flights'][0]['航班号']=='CZ1111'
    assert b.evaluate('FlightJobs.selected()')==br['job_id']
    b.goto(base+'/overview')
    b.evaluate("goToTimeline('CZ2222')")
    b.wait_for_url('**/timeline?*')
    assert br['job_id'] in b.url
    a.screenshot(path=str(Path(__file__).resolve().parents[2]/'reports/phase1-test-ui.png'),full_page=True)

def test_scenario_values_are_rendered_as_text(browser):
    context,base=browser
    page=context.new_page()
    name='<img src=x onerror=window.__scenarioXss=1>'
    page.goto(base+'/flight_input')
    prediction=page.evaluate('''async flight => (await fetch('/predict_flight', {
      method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(flight)})).json()''',FLIGHT)
    page.evaluate('''() => document.body.appendChild(createFlightResultPlaceholder(99, {
      '机尾号':'<img src=x onerror=window.__flightXss=1>','航班号':'CZ0001'}))''')
    assert page.evaluate('window.__flightXss') is None
    response=page.request.post(base+'/api/save-scenario',data={
        'scenario_name':name,'stop_periods':[{'start_time':'06:00','end_time':'07:00'}],
        'hourly_capacity':10})
    assert response.ok
    page.goto(base+'/scenario')
    page.locator('.scenario-item').first.wait_for()
    assert page.evaluate('window.__scenarioXss') is None
    assert page.locator('.scenario-name').first.text_content()==name
    page.locator('.scenario-item .apply-btn').first.click()
    page.wait_for_url('**/timeline?*')
    assert prediction['job_id'] in page.url
    page.locator('.timeline-section').wait_for()
    assert page.evaluate('window.__scenarioXss') is None
    assert name in page.locator('.timeline-section').text_content()

def test_sse_parser_handles_utf8_split_at_every_byte(browser):
    context,base=browser
    page=context.new_page();page.goto(base+'/flight_input')
    result=page.evaluate('''async () => {
      const bytes=new TextEncoder().encode('data: '+JSON.stringify({message:'航班预测'})+'\\n\\n'+'data: '+JSON.stringify({type:'complete'})+'\\n\\n');
      const response=new Response(new ReadableStream({start(c){for(const byte of bytes)c.enqueue(new Uint8Array([byte]));c.close();}}));
      const items=[];for await(const item of FlightJobs.events(response))items.push(item);return items;
    }''')
    assert result==[{'message':'航班预测'},{'type':'complete'}]
