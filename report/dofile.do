
// Global Path

			if "`c(username)'"=="ogazmuri"  {                        
					global main_dir = "C:\Users\ogazmuri\Documents\Git\QueVasaEstudiar" 		
					global main_analysis = "C:\Users\ogazmuri\Google Drive\ProyectoICFES-Bot\Worked"
			} 

			if "`c(username)'"=="obord"  {                        
					global main_dir = "C:\Users\obord\Documents\Git\QueVasaEstudiar" 		
					global main_analysis = "C:\Users\obord\Google Drive\ProyectoICFES-Bot\Worked"
			} 

			if "`c(username)'"=="Franco"  {                        
					global main_dir = "C:\Users\Franco\GitHub\QueVasaEstudiar" 		
					global main_analysis = "D:\Google Drive\ProyectoICFES-Bot\Worked"
			} 


cd "${main_dir}"


***********************************************************
/*
		II.1 Firebase: Interactions

		We can transform the times date(inalambria) = date(firebase) - 5 horas
*/
***********************************************************


import delimited using "interactions.txt", delim(";") varnames(1)  clear 

drop if user=="default-user"
drop if length(user)<20

rename user url_id

gen date_envio=substr(timestamp,1,strpos(timestamp,"T")-1)

preserve
	import delimited using "Student_Features_Fall2018.csv", delim(",") case(preserve) varnames(1) clear 
	tempfile aux
	save `aux'
restore

merge m:1 url_id using `aux', keep(3) nogen

tab event_name

rename CELULAR celular

replace celular=subinstr(celular,"(","",.)
replace celular=subinstr(celular,")","",.)
replace celular=subinstr(celular," ","",.)
replace celular=subinstr(celular,"-","",.)
replace celular="57"+celular
destring celular, replace

gen time_envio=substr(timestamp,strpos(timestamp,"T")+1,.)
gen hour_envio = substr(time_envio,1,2)
gen minute_envio = substr(substr(time_envio,4,5),1,2)
gen second_envio = substr(substr(time_envio,7,8),1,2)

destring *, replace
***Transform to Colombian time (server is in another time zome)

replace hour_envio = hour_envio - 5
replace hour_envio = hour_envio + 24 if hour_envio<0

// some times when the messages are over X characters, they get reported as several
bys celular date_envio hour_envio minute_envio second_envio (timestamp): keep if _n==1


egen id = group(celular date_envio hour_envio minute_envio second_envio)

save "${main_analysis}\Interactions.dta", replace


// Generate file with only Menu interactions for Franco
keep if event_name=="OPTIONS" | event_name=="OPTIONS_SELECTION"
bys Student_ID (timestamp): gen Selection = info[_n+1] if event_name=="OPTIONS" & event_name[_n+1]=="OPTIONS_SELECTION"

drop if event_name=="OPTIONS_SELECTION"

drop event_name

rename info Menu

order Selection, after(Menu)

save "${main_analysis}\Menus_interactions.dta", replace
***********************************************************
/*
		III. Cool Graphs, for report
*/
***********************************************************
set scheme s1color

import delimited using "Student_Features_Fall2018.csv", delim(",") case(preserve) varnames(1) clear 

rename CELULAR celular
replace celular=subinstr(celular,"(","",.)
replace celular=subinstr(celular,")","",.)
replace celular=subinstr(celular," ","",.)
replace celular=subinstr(celular,"-","",.)
replace celular="57"+celular
destring celular, replace
keep Student_ID celular url_id day_group

merge 1:m url_id using "${main_analysis}\Interactions.dta", keepusing(event_name hour_envio) keep(1 3)

bys url_id: gen N_interactions = _N
replace N_interactions=. if _m==1

gen menu=(event_name=="OPTIONS")
bys url_id: egen N_menu=sum(menu)
replace N_menu=. if _m==1
drop event_name menu

gen Interacted = (_m==3)
drop _m
drop if !(day_group == "D01" | day_group == "D02" | day_group == "D03") 
gen N=1

replace N_menu = . if  N_interactions > 1000
replace Interacted = 0 if  N_interactions > 1000
replace N_interactions = . if  N_interactions > 1000

preserve
	// First set of graphs, by hour
	collapse (mean) Interacted N_interactions N_menu (sum) N, by(hour_envio)

	label var hour_envio "Hora Envío"
	label var Interacted "% Interactuó"
	label var N_menu "# Promedio de menus"
	label var N_interactions "# Promedio de interacciones"

	line Interacted hour_envio, ylabel(0 (0.02) 1, grid) xlabel(0(1)23)
	graph export "${main_analysis}\Graphs\Interaction_hourly.eps", as(eps) replace

	line N_interactions hour_envio, ylabel(0 (30) 240, grid) xlabel(0(1)23) xtitle(Time) ytitle(Average number of interactions)

	line N_menu hour_envio, ylabel(0 (1) 16, grid) xlabel(0(1)23) xtitle(Time) ytitle(Average number of menus)
	graph export "${main_analysis}\Graphs\N_hourly.eps", as(eps) replace
restore
