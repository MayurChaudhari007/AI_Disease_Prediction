document.addEventListener("DOMContentLoaded", function () {
    const printBtn = document.getElementById("downloadPDF");

    if (printBtn) {
        printBtn.addEventListener("click", function (e) {
            e.preventDefault(); // stop # jump / form submit

            let printContents = document.getElementById("report").innerHTML;
            let originalContents = document.body.innerHTML;

            document.body.innerHTML = printContents;
            window.print();
            document.body.innerHTML = originalContents;
        });
    }
});