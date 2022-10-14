def run_sweetviz_report(df, TARGET):
    
    import sweetviz as sv
    from datetime import datetime
    
    report_label = datetime.today().strftime('%Y-%m-%d_%H_%M')
    
    my_report = sv.analyze(df,target_feat=TARGET)
    my_report.show_html(filepath='SWEETVIZ_' + report_label + '.html')
    
    return